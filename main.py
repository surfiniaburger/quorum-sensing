# main.py (or your relevant agent definition file)

import asyncio
import contextlib
from contextlib import asynccontextmanager
import json
import hashlib
from typing import Any, Dict, List, Optional, Tuple, AsyncGenerator, ClassVar
from typing_extensions import override
from fastapi import HTTPException
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
# Core ADK Agent imports
from google.adk.agents import LlmAgent, BaseAgent, LoopAgent, SequentialAgent, ParallelAgent # Added ParallelAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.tools.agent_tool import AgentTool
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters
from google.genai import types
from google.adk.models.lite_llm import LiteLlm
from google.adk.events import Event
from pydantic import BaseModel, Field
from starlette.websockets import WebSocketDisconnect
from voice_agent import router as voice_agent_router # Assuming this exists
import hashlib # For task ID generation if not already there
from google.adk.tools import LongRunningFunctionTool
from adk_native.long_running_image_ops import initiate_image_generation, IMAGE_GENERATION_TASKS
from adk_native.long_running_video_ops import initiate_video_generation, VIDEO_GENERATION_TASKS
from adk_native.long_running_audio_ops import (
    initiate_tts_generation, TTS_GENERATION_TASKS,
    initiate_stt_transcription, STT_TRANSCRIPTION_TASKS
)
import logging


# --- MODIFIED LOGGING CONFIGURATION ---
LOG_FILE_NAME = "app.log" # Define your log file name

# Configure basic console logging (as before)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)

# Get the root logger (which basicConfig configures)
root_logger = logging.getLogger()

# Create a file handler to write logs to a file
# For long-running applications, consider using logging.handlers.RotatingFileHandler
# from logging.handlers import RotatingFileHandler
# file_handler = RotatingFileHandler(LOG_FILE_NAME, maxBytes=10*1024*1024, backupCount=5) # e.g., 10MB per file, 5 backups
file_handler = logging.FileHandler(LOG_FILE_NAME)
file_handler.setLevel(logging.INFO) # Set the logging level for the file

# Create a formatter and set it for the file handler (same format as console)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
file_handler.setFormatter(file_formatter)

# Add the file handler to the root logger
# Now logs will go to both console (from basicConfig) and the file
root_logger.addHandler(file_handler)

# The application's main logger instance will use this root logger's configuration
logger = logging.getLogger(__name__) # Use this for consistency
# --- END OF MODIFIED LOGGING CONFIGURATION ---
# --- Configuration & Global Setup ---


# --- Configuration & Global Setup (Copied from your existing code) ---
load_dotenv()

APP_NAME = "ADK_MCP_App_Parallel" # Changed to reflect new structure
MODEL_ID = "gemini-2.0-flash"
GEMINI_PRO_MODEL_ID = "gemini-2.5-pro-preview-05-06"
GEMINI_FLASH_MODEL_ID="gemini-2.5-flash-preview-04-17" # Ensure correct new name
STATIC_DIR = "static"

session_service = InMemorySessionService()
artifacts_service = InMemoryArtifactService()
loaded_mcp_tools_global: Dict[str, Any] = {}

class AllServerConfigs(BaseModel):
    configs: Dict[str, StdioServerParameters]

# --- Server Parameter Definitions (Copied) ---
weather_server_params = StdioServerParameters(command="python", args=["./mcp_server/weather_server.py"])
ct_server_params = StdioServerParameters(command="python", args=["./mcp_server/cocktail.py"])
bnb_server_params = StdioServerParameters(command="npx", args=["-y", "@openbnb/mcp-server-airbnb", "--ignore-robots-txt"])
mlb_stats_server_params = StdioServerParameters(command="python", args=["./mcp_server/mlb_stats_server.py"])
web_search_server_params = StdioServerParameters(command="python", args=["./mcp_server/web_search_server.py"])
bq_vector_search_server_params = StdioServerParameters(command="python", args=["./mcp_server/bq_vector_search_server.py"])
visual_asset_server_params = StdioServerParameters(command="python", args=["./mcp_server/visual_asset_server.py"])
static_retriever_server_params = StdioServerParameters(command="python", args=["./mcp_server/static_asset_retriever_mcp_server.py"])
image_embedding_server_params = StdioServerParameters(command="python", args=["./mcp_server/image_embedding_server.py"])
video_clip_server_params = StdioServerParameters(command="python", args=["./mcp_server/video_clip_server.py"])
audio_processing_server_params = StdioServerParameters(command="python", args=["./mcp_server/audio_processing_server.py"])
game_plotter_tool_params = StdioServerParameters(command="python", args=["./mcp_server/game_plotter_mcp_server.py"])

server_configs_instance = AllServerConfigs(
    configs={
        "weather": weather_server_params, "bnb": bnb_server_params, "ct": ct_server_params,
        "mlb": mlb_stats_server_params, "web_search": web_search_server_params,
        "bq_search": bq_vector_search_server_params, "visual_assets": visual_asset_server_params,
        "static_retriever_mcp": static_retriever_server_params,
        "image_embedding_mcp": image_embedding_server_params,
        "video_clip_generator_mcp": video_clip_server_params,
        "audio_processing_mcp": audio_processing_server_params,
        "game_plotter_tool": game_plotter_tool_params
    }
)


# --- Define LongRunningFunctionTool instances ---
long_running_image_tool = LongRunningFunctionTool(func=initiate_image_generation)
long_running_video_tool = LongRunningFunctionTool(func=initiate_video_generation)
long_running_tts_tool = LongRunningFunctionTool(func=initiate_tts_generation)
long_running_stt_tool = LongRunningFunctionTool(func=initiate_stt_transcription)



# --- Agent Instructions (Copied) ---
ROOT_AGENT_INSTRUCTION = """
**Role:** You are a Virtual Assistant acting as a Request Router.
**Primary Goal:** Analyze user requests and route them to the correct specialist sub-agent.
**Capabilities & Routing:**
* **Greetings:** If the user greets you, respond warmly and directly.
* **Cocktails:** Route requests about cocktails, drinks, recipes, or ingredients to `cocktail_assistant`.
* **Booking & Weather:** Route requests about booking accommodations or weather to `booking_assistant`.
* **MLB Information (General):** Route general requests concerning Major League Baseball (MLB) stats, scores, schedules, rosters, standings to the `mlb_assistant`.
    The `mlb_assistant` will handle obtaining any necessary IDs (like `game_pk`, `player_id`, `team_id`) if not provided by the user for these general queries.
* **MLB Game Recap:** If the user specifically asks for a "game recap", "recap of the game", "game summary" or similar, and a specific game can be identified (e.g., "recap of yesterday's Yankees game" or "recap for game PK 12345"), route the request to the `game_recap_assistant_v2`.
    - If a `game_pk` is mentioned or easily derivable from the query (e.g. from a team name and date like "yesterday's Yankees game"), include it or the identifying information in the routing.
    - If the game for the recap is unclear, you can first delegate to `mlb_assistant` to help identify the `game_pk`, and then if `game_pk` is found, the user might be prompted to ask for the recap again, or you could try re-routing. (Simpler: for now, assume if routed to `game_recap_assistant_v2`, the query contains enough info to derive game_pk, or `game_recap_assistant_v2` will handle clarification if needed).
* **Out-of-Scope:** If the request is unrelated, state directly that you cannot assist.
**Key Directives:**
* **Delegate Immediately:** Once a suitable sub-agent is identified, route the request.
* **Do Not Answer Delegated Topics:** You must **not** attempt to answer questions for delegated topics yourself.
* **Formatting:** Format your final response using Markdown.
* **Game Recap Clarification:** If a user asks for a game recap but doesn't specify which game, ask them to specify the game (e.g., "Which game would you like a recap for? Please provide the teams and date, or the game ID if you know it.") before attempting to route to `game_recap_assistant_v2`. If they provide details, then route.
""" # Note: Changed game_recap_assistant to game_recap_assistant_v2 for the new parallel version

# --- Helper Function (Copied) ---
def _clean_json_string_from_llm(raw_json_string: Optional[str], default_if_empty: str = "[]") -> str:
    if not raw_json_string: return default_if_empty
    clean_string = raw_json_string.strip()
    if clean_string.startswith("```json"):
        clean_string = clean_string[7:]
        if clean_string.endswith("```"): clean_string = clean_string[:-3]
    elif clean_string.startswith("```"):
        clean_string = clean_string[3:]
        if clean_string.endswith("```"): clean_string = clean_string[:-3]
    return clean_string.strip() if clean_string.strip() else default_if_empty


# This class definition goes at the top level of your main.py, with other class definitions.
class GameStateSetterAgent(BaseAgent):
    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, name: str): # No default_query needed here
        super().__init__(name=name, sub_agents=[])

    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        logger.info(f"[{self.name}] Starting: Attempting to set game_pk from InitialRecapGenerator's output.")
        initial_output_text = ctx.session.state.get("current_recap") # Output from InitialRecapGenerator

        identified_game_pk = None
        cleaned_recap_text = initial_output_text # Default to original if PK line not found or no cleaning needed

        if initial_output_text:
            import re
            # Look for the specific marker line
            match = re.search(r"^INTERNAL_GAME_PK_FOUND:\s*(\S+)\s*$", initial_output_text, re.MULTILINE | re.IGNORECASE)
            if match:
                pk_value_str = match.group(1)
                if pk_value_str.upper() == "NONE":
                    identified_game_pk = None
                    logger.info(f"[{self.name}] InitialRecapGenerator indicated no game_pk (INTERNAL_GAME_PK_FOUND: NONE).")
                else:
                    identified_game_pk = pk_value_str
                    logger.info(f"[{self.name}] Parsed game_pk: '{identified_game_pk}' using INTERNAL_GAME_PK_FOUND marker.")
                
                # Option: Remove the marker line from the recap text that gets passed on
                # This makes "current_recap" cleaner for subsequent dialogue agents.
                # Split by lines, filter out the marker, and rejoin.
                lines = initial_output_text.splitlines()
                cleaned_lines = [line for line in lines if not re.match(r"^INTERNAL_GAME_PK_FOUND:\s*\S+\s*$", line, re.IGNORECASE)]
                cleaned_recap_text = "\n".join(cleaned_lines).strip()
                if initial_output_text != cleaned_recap_text:
                     logger.info(f"[{self.name}] Cleaned INTERNAL_GAME_PK_FOUND marker from current_recap.")

            else:
                logger.warning(f"[{self.name}] Could not find 'INTERNAL_GAME_PK_FOUND:' marker in InitialRecapGenerator's output. 'game_pk' will be set to None by this agent.")
                # Ensure 'game_pk' is explicitly None if not found by marker
                identified_game_pk = None # Redundant if already None, but safe

            ctx.session.state["game_pk"] = str(identified_game_pk) if identified_game_pk is not None else None
            ctx.session.state["current_recap"] = cleaned_recap_text # Store the (potentially cleaned) recap

        else:
            logger.warning(f"[{self.name}] 'current_recap' from InitialRecapGenerator is empty or None. 'game_pk' will be set to None.")
            ctx.session.state["game_pk"] = None
            ctx.session.state["current_recap"] = None # Ensure recap is also None if input was None

        thought_text = f"GameStateSetter: Processed initial output. Game PK in session state is now: {ctx.session.state.get('game_pk')}. Recap text updated (if marker was present)."
        logger.info(f"[{self.name}] {thought_text}")
        yield Event(
            author=self.name,
            content=types.Content(
                role="model",
                parts=[types.Part(text=thought_text)]
            )
        )
        logger.info(f"[{self.name}] Finished processing.")




class QuerySetupAgent(BaseAgent):
    model_config = {"arbitrary_types_allowed": True}

    default_query: str = "generic recap request"

    def __init__(self, name: str, default_query: Optional[str] = None, **kwargs):
        init_kwargs = {
            "name": name,
            "sub_agents": [],
            **kwargs
        }
        if default_query is not None:
            init_kwargs['default_query'] = default_query
        
        super().__init__(**init_kwargs)

    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        logger.info(f"[{self.name}] Setting up 'user_query' in session state. Agent's default_query is: '{self.default_query}'")

        current_user_input_text = None
        # Access history via ctx.session.events
        if ctx.session and ctx.session.events:  # Check if session and events list exist
            logger.info(f"[{self.name}] Accessing ctx.session.events for user_query. History length: {len(ctx.session.events)}")
            for event in reversed(ctx.session.events): # Iterate through the session's event history
                if event.author == "user" and event.content and event.content.parts:
                    # Ensure part is not None and has text
                    if event.content.parts[0] and hasattr(event.content.parts[0], 'text'):
                        text_part_content = event.content.parts[0].text
                        if text_part_content: # Ensure text is not empty
                            current_user_input_text = text_part_content
                            logger.info(f"[{self.name}] Found user message in history: '{current_user_input_text}'")
                            break
            if not current_user_input_text:
                logger.info(f"[{self.name}] No user message with text found in ctx.session.events.")
        else:
            logger.info(f"[{self.name}] ctx.session or ctx.session.events is None/empty. Cannot get user_query from history.")
        
        effective_user_query = ""
        if current_user_input_text:
            effective_user_query = current_user_input_text
            logger.info(f"[{self.name}] 'user_query' will be set from current user input in history: '{current_user_input_text}'")
        else:
            # Fallback: Check if 'user_query' was somehow already set in the state (e.g., by an earlier process or initial state)
            user_query_from_state = ctx.session.state.get("user_query")
            if user_query_from_state:
                effective_user_query = user_query_from_state
                logger.info(f"[{self.name}] 'user_query' already present in state: '{user_query_from_state}'")
            else:
                effective_user_query = self.default_query # Use the Pydantic-managed instance attribute
                logger.info(f"[{self.name}] 'user_query' was not found in history or state, will be set to agent's default: '{self.default_query}'")
        
        ctx.session.state["user_query"] = effective_user_query

        if "player_lookup_dict_json" not in ctx.session.state:
            ctx.session.state["player_lookup_dict_json"] = "{}"
            logger.info(f"[{self.name}] Initialized 'player_lookup_dict_json' to empty JSON object string.")
        if "revision_number" not in ctx.session.state:
            ctx.session.state["revision_number"] = 0 # Or 1 if you prefer 1-based
        logger.info(f"[{self.name}] Initialized 'revision_number' to {ctx.session.state['revision_number']}.")

        yield Event(
            author=self.name,
            content=types.Content(role="model", parts=[types.Part(text=f"QuerySetupAgent: user_query is now '{ctx.session.state.get('user_query')}'")])
        )
        logger.info(f"[{self.name}] Finished. user_query in state: {ctx.session.state.get('user_query')}")



# REVISED DirectToolCallerBaseAgent for better FunctionResponse parsing

# In main.py

class DirectToolCallerBaseAgent(BaseAgent):
    # Pydantic model_config:
    # "extra = 'ignore'" (default) means undeclared fields in __init__ are ignored.
    # "extra = 'allow'" means undeclared fields are allowed and stored on the model.
    # BaseAgent might have its own model_config. Let's assume it allows additional attributes
    # or we can set "extra = 'allow'" if needed, though usually not required for attributes
    # set *after* super().__init__ that are not meant to be Pydantic fields.
    model_config = {"arbitrary_types_allowed": True, "extra": "allow"} # Let's be explicit

    # These are Pydantic fields, correctly declared:
    tool_name_to_call: str
    input_state_keys: Dict[str, str]
    output_state_key: str
    default_output_on_error: str
    fixed_tool_args: Dict[str, Any] = {}

    # _internal_llm will NOT be a Pydantic field.
    # It will be a regular instance attribute.
    # Remove: _internal_llm: Optional[LlmAgent] = Field(None, exclude=True)

    def __init__(self,
                 name: str,
                 tool_name_to_call: str,
                 input_state_keys: Dict[str, str],
                 output_state_key: str,
                 default_output_on_error: str = "{}",
                 fixed_tool_args: Optional[Dict[str, Any]] = None,
                 **kwargs):
        super_kwargs = {
            "name": name,
            "tool_name_to_call": tool_name_to_call,
            "input_state_keys": input_state_keys,
            "output_state_key": output_state_key,
            "default_output_on_error": default_output_on_error,
            "fixed_tool_args": fixed_tool_args or {},
            **kwargs
        }
        super().__init__(**super_kwargs)

        # Initialize _internal_llm as a standard Python instance attribute
        # AFTER super().__init__() has completed.
        # Pydantic will not treat this as one of its fields.
        self._internal_llm: Optional[LlmAgent] = None

    def _get_or_initialize_internal_llm(self, ctx: InvocationContext) -> Optional[LlmAgent]:
        if self._internal_llm is None: # Access the instance attribute
            tool_instance_to_use = None
            try:
                tool_instance_to_use = ctx.get_tool(self.tool_name_to_call)
            except Exception as e:
                logger.debug(f"[{self.name}] ctx.get_tool for '{self.tool_name_to_call}' failed: {e}. Trying global.")
            
            if not tool_instance_to_use:
                toolset_name, func_name = self.tool_name_to_call.split('.', 1) if '.' in self.tool_name_to_call else (None, None)
                if toolset_name and func_name and toolset_name in loaded_mcp_tools_global:
                    for t in loaded_mcp_tools_global[toolset_name]:
                        tool_name_attr = getattr(t, 'name', None)
                        decl_name_attr = getattr(t, 'function_declaration', None)
                        if tool_name_attr == func_name or \
                           (decl_name_attr and decl_name_attr.name == self.tool_name_to_call):
                            tool_instance_to_use = t
                            break
                if not tool_instance_to_use:
                    logger.error(f"[{self.name}] Could not find tool instance for '{self.tool_name_to_call}'.")
                    return None
            
            temp_args_state_key = f"temp_tool_args_for_{self.name}_{self.tool_name_to_call.replace('.', '_')}"
            instruction = f"""
our ONLY task is to make a SINGLE call to the tool named '{self.tool_name_to_call}'.
All arguments for this tool call are in a dictionary in session state under the key '{temp_args_state_key}'.
You MUST use these arguments precisely.
The tool named '{self.tool_name_to_call}' will execute and provide its complete, raw JSON string output directly to you as part of its function response.
Your entire and final textual output for this interaction MUST BE this exact, verbatim JSON string that the tool provided.
For example, if the tool returns the JSON string `"[{{\\"uri\\": \\"gs://example.png\\"}}]"` then your output MUST be exactly that string: `"[{{\\"uri\\": \\"gs://example.png\\"}}]"`
DO NOT summarize, explain, add "Okay", or change the tool's JSON string output in any way.
If the tool call results in an error being returned by the tool itself (as a JSON error string), your output should be that error JSON string.
If the input arguments in '{temp_args_state_key}' are missing or clearly invalid for the tool, your output MUST be the exact JSON string: "{self.default_output_on_error}".
"""

            # Initialize the instance attribute _internal_llm
            self._internal_llm = LlmAgent(
                name=f"{self.name}_InternalToolCaller", model=GEMINI_PRO_MODEL_ID,
                instruction=instruction, tools=[tool_instance_to_use],
                output_key=f"temp_result_for_{self.name}_{self.tool_name_to_call.replace('.', '_')}"
            )
            logger.info(f"[{self.name}] Initialized internal LlmAgent for '{self.tool_name_to_call}'.")
        return self._internal_llm # Return the instance attribute

    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        # ... (The rest of this method remains the same as your last working version,
        #      it will use self._get_or_initialize_internal_llm() which accesses self._internal_llm) ...
        logger.info(f"[{self.name}] Orchestrating tool call for '{self.tool_name_to_call}'.")
        tool_args = {}
        tool_args.update(self.fixed_tool_args)
        for tool_arg_name, state_key in self.input_state_keys.items():
            value = ctx.session.state.get(state_key)
            if value is not None: tool_args[tool_arg_name] = value
            elif tool_arg_name not in tool_args:
                 logger.warning(f"[{self.name}] Input state key '{state_key}' for '{tool_arg_name}' is None and not in fixed_args.")

        if "prompts" in tool_args and isinstance(tool_args["prompts"], str) and \
           (self.tool_name_to_call == "visual_assets.generate_images_from_prompts" or \
            self.tool_name_to_call == "video_clip_generator_mcp.generate_video_clips_from_prompts"):
            try:
                tool_args["prompts"] = json.loads(_clean_json_string_from_llm(tool_args["prompts"], default_if_empty="[]"))
                logger.info(f"[{self.name}] Converted 'prompts' from JSON string to list for tool call.")
            except json.JSONDecodeError:
                logger.error(f"[{self.name}] Failed to parse 'prompts' JSON: {tool_args['prompts']}.")
                ctx.session.state[self.output_state_key] = self.default_output_on_error
                yield Event(author=self.name, content=types.Content(role="model", parts=[types.Part(text=f"{self.name}: Failed to parse input prompts.")]))
                return

        internal_llm = self._get_or_initialize_internal_llm(ctx) # Uses the instance attribute
        if not internal_llm:
            logger.error(f"[{self.name}] Failed to initialize internal LlmAgent. Cannot call tool.")
            ctx.session.state[self.output_state_key] = self.default_output_on_error
            yield Event(author=self.name, content=types.Content(role="model", parts=[types.Part(text=f"{self.name}: Internal error during setup.")]))
            return

        temp_args_state_key = f"temp_tool_args_for_{self.name}_{self.tool_name_to_call.replace('.', '_')}"
        temp_result_state_key = internal_llm.output_key

        ctx.session.state[temp_args_state_key] = tool_args
        logger.info(f"[{self.name}] Running internal LlmAgent. Args in '{temp_args_state_key}', result to '{temp_result_state_key}'.")

        async for event in internal_llm.run_async(ctx): yield event

        raw_tool_payload_str = ctx.session.state.get(temp_result_state_key, self.default_output_on_error)
        tool_payload_str = raw_tool_payload_str

        tool_call_successful = True
        if isinstance(tool_payload_str, str):
            try:
                parsed_check = json.loads(tool_payload_str)
                if isinstance(parsed_check, dict) and "error" in parsed_check:
                    tool_call_successful = False
                    logger.warning(f"[{self.name}] Tool call returned an error JSON: {tool_payload_str}")
            except json.JSONDecodeError:
                if tool_payload_str != self.default_output_on_error or \
                   (self.default_output_on_error == "{}" and tool_payload_str != "{}") or \
                   (self.default_output_on_error == "[]" and tool_payload_str != "[]"):
                    logger.error(f"[{self.name}] Internal LlmAgent for '{self.tool_name_to_call}' output non-JSON: {tool_payload_str[:200]}...")
                    tool_call_successful = False
        else:
            tool_call_successful = False
            logger.error(f"[{self.name}] Expected string from internal LlmAgent ('{temp_result_state_key}'), got {type(tool_payload_str)}.")
            tool_payload_str = self.default_output_on_error
        
        ctx.session.state[self.output_state_key] = tool_payload_str
        if temp_args_state_key in ctx.session.state: del ctx.session.state[temp_args_state_key]
        if temp_result_state_key in ctx.session.state: del ctx.session.state[temp_result_state_key]

        final_thought = f"{self.name}: Tool '{self.tool_name_to_call}' processed. Output to '{self.output_state_key}'. Success: {tool_call_successful}. Payload: {str(tool_payload_str)[:100]}..."
        logger.info(final_thought)
        yield Event(author=self.name, content=types.Content(role="model", parts=[types.Part(text=final_thought)]))
        
################################################################################
# ---  PlayerIdPopulatorAgent ---
################################################################################

class PlayerIdPopulatorAgent(BaseAgent):
    model_config = {"arbitrary_types_allowed": True} 

    # This is the Pydantic field that will hold the tools.
    # It's a required field because it doesn't have a default value.
    mlb_tools_for_roster_fetcher: List # Expects a list of tool instances

    # Internal LlmAgent instance, not a Pydantic field managed at BaseAgent init
    roster_fetcher_llm: Optional[LlmAgent] = Field(None, exclude=True) # Exclude from Pydantic model if it's purely internal state

    def __init__(self, name: str, mlb_tools: List, **kwargs):
        # We receive 'mlb_tools' from the instantiation call.
        # We need to pass this to super().__init__ using the keyword
        # that matches the declared field name 'mlb_tools_for_roster_fetcher'.

        # Add 'mlb_tools_for_roster_fetcher' to kwargs that will be passed to super
        # This ensures Pydantic initializes the declared field.
        super_kwargs = {
            "name": name,
            "mlb_tools_for_roster_fetcher": mlb_tools, # Key matches the declared field
            **kwargs  # Pass through any other kwargs BaseAgent might expect or Pydantic handles
        }
        super().__init__(**super_kwargs)

        # self.mlb_tools_for_roster_fetcher is now set by Pydantic.
        # Initialize other non-Pydantic instance variables here if needed (like _roster_fetcher_llm)
        # self._roster_fetcher_llm is already defaulted to None at class level via Field.

    def _get_or_initialize_roster_fetcher_llm(self) -> Optional[LlmAgent]:
        if self.roster_fetcher_llm is None:
            # Access the Pydantic-managed field 'self.mlb_tools_for_roster_fetcher'
            if not self.mlb_tools_for_roster_fetcher:
                 logger.error(f"[{self.name}] MLB tools (mlb_tools_for_roster_fetcher) list is empty or was not properly initialized!")
                 return None

            logger.info(f"[{self.name}] Initializing internal RosterFetcherLlm with {len(self.mlb_tools_for_roster_fetcher)} MLB tools.")
            self.roster_fetcher_llm = LlmAgent(
                name=f"{self.name}_RosterFetcherLlm",
                model=GEMINI_FLASH_MODEL_ID, # Or your preferred model
                instruction="""You are a sports data assistant.
You will be given a list of MLB team names in session state via the key `teams_for_roster_lookup` (e.g., ["Houston Astros", "Seattle Mariners"]).
1. For each team name in the list, you MUST call the `mlb.get_team_roster` tool.
   - Use the team name as the `team_identifier`.
   - Use the current year (e.g., 2025, or derive if possible, otherwise assume a recent common year like 2024 or 2025) for the `season` parameter.
   - Use a common `roster_type` like "40Man" or "active" (e.g., "40Man").
2. Collect all player data from all successfully fetched rosters. From each player entry, extract their full name (e.g., from a "fullName" field) and their player ID (e.g., from a "person.id" field, ensuring it's treated as a string).
3. Your final output for this entire task MUST be a single JSON string. This JSON string must represent a dictionary where:
   - Keys are player full names.
   - Values are their player IDs (as strings).
   Example output: `{"Jose Altuve": "514888", "Yordan Alvarez": "670541"}`.
4. If a team name is invalid or a roster is not found, omit players from that team. If no rosters are found for any team, or if the input list of teams is empty, output an empty JSON object string: `{}`.
DO NOT add any conversational text. Output ONLY the JSON string map.
                """,
                tools=self.mlb_tools_for_roster_fetcher, # Correctly uses the instance's tools
                output_key="temp_player_id_map_json"
            )
        return self.roster_fetcher_llm

    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        # ... (This method's logic remains the same as the last correct version)
        logger.info(f"[{self.name}] Starting: Populating player ID lookup dictionary.")
        
        roster_fetcher = self._get_or_initialize_roster_fetcher_llm()

        if not roster_fetcher:
            logger.error(f"[{self.name}] RosterFetcherLlm could not be initialized. Skipping player ID population.")
            ctx.session.state["player_lookup_dict_json"] = "{}"
            yield Event(author=self.name, content=types.Content(role="model", parts=[types.Part(text="PlayerIdPopulator: Skipped due to internal LlmAgent init error.")]))
            return

        extracted_teams_str = ctx.session.state.get("extracted_teams_for_roster_json")
        team_names_for_roster_lookup = []

        if extracted_teams_str:
            try:
                cleaned_extracted_teams_str = _clean_json_string_from_llm(extracted_teams_str, default_if_empty="{}")
                entities = json.loads(cleaned_extracted_teams_str)
                if isinstance(entities, dict) and "teams" in entities and isinstance(entities["teams"], list):
                    team_names_for_roster_lookup = [str(team) for team in entities["teams"] if team]
                    logger.info(f"[{self.name}] Found teams for roster lookup: {team_names_for_roster_lookup}")
                else:
                    logger.warning(f"[{self.name}] 'extracted_teams_for_roster_json' was not a dict with a 'teams' list. JSON: {cleaned_extracted_teams_str}")
            except json.JSONDecodeError:
                logger.error(f"[{self.name}] Failed to parse 'extracted_teams_for_roster_json': {extracted_teams_str}")
        
        if not team_names_for_roster_lookup:
            logger.warning(f"[{self.name}] No team names for roster lookup. 'player_lookup_dict_json' will be empty.")
            ctx.session.state["player_lookup_dict_json"] = "{}"
            yield Event(author=self.name, content=types.Content(role="model", parts=[types.Part(text="PlayerIdPopulator: No teams identified.")]))
            return

        ctx.session.state["teams_for_roster_lookup"] = team_names_for_roster_lookup
        logger.info(f"[{self.name}] Running RosterFetcherLlm for teams: {team_names_for_roster_lookup}")
        async for event in roster_fetcher.run_async(ctx): yield event

        player_id_map_json_str = ctx.session.state.get("temp_player_id_map_json", "{}")
        cleaned_player_id_map_json_str = _clean_json_string_from_llm(player_id_map_json_str, default_if_empty="{}")
        
        final_player_lookup = {}
        try:
            parsed_map = json.loads(cleaned_player_id_map_json_str)
            if isinstance(parsed_map, dict):
                final_player_lookup = {str(name): str(pid) for name, pid in parsed_map.items() if name and pid is not None}
                logger.info(f"[{self.name}] Processed player ID map. Count: {len(final_player_lookup)}")
            else:
                logger.warning(f"[{self.name}] RosterFetcherLlm output not a dict: {cleaned_player_id_map_json_str}")
        except json.JSONDecodeError:
            logger.error(f"[{self.name}] Failed to parse RosterFetcherLlm output: {cleaned_player_id_map_json_str}")

        ctx.session.state["player_lookup_dict_json"] = json.dumps(final_player_lookup)
        if "teams_for_roster_lookup" in ctx.session.state: del ctx.session.state["teams_for_roster_lookup"]
        if "temp_player_id_map_json" in ctx.session.state: del ctx.session.state["temp_player_id_map_json"]

        thought_text = f"PlayerIdPopulator: 'player_lookup_dict_json' now contains {len(final_player_lookup)} entries."
        logger.info(f"[{self.name}] {thought_text}")
        yield Event(author=self.name, content=types.Content(role="model", parts=[types.Part(text=thought_text)]))

################################################################################
# --- PHASE 1: StaticAssetPipelineAgent (Copied - Assumed Correct) ---
################################################################################
class StaticAssetPipelineAgent(BaseAgent):
    model_config = {"arbitrary_types_allowed": True}
    entity_extractor: LlmAgent
    static_asset_query_generator: LlmAgent
    logo_searcher_llm: DirectToolCallerBaseAgent
    headshot_retriever_llm: DirectToolCallerBaseAgent

    def __init__(self, name: str,
                 entity_extractor: LlmAgent,
                 static_asset_query_generator: LlmAgent,
                 logo_searcher_llm: DirectToolCallerBaseAgent,
                 headshot_retriever_llm: DirectToolCallerBaseAgent):
        super().__init__(
            name=name,
            entity_extractor=entity_extractor,
            static_asset_query_generator=static_asset_query_generator,
            logo_searcher_llm=logo_searcher_llm,
            headshot_retriever_llm=headshot_retriever_llm,
            sub_agents=[entity_extractor, static_asset_query_generator, logo_searcher_llm, headshot_retriever_llm]
        )

    def _extract_team_name_from_query(self, query_string: str) -> Optional[str]:
        query_lower = query_string.lower()
        if " logo" in query_lower:
            team_name = query_lower.replace(" logo", "").strip()
            return ' '.join(word.capitalize() for word in team_name.split())
        return None

    def _extract_player_name_from_query(self, query_string: str) -> Optional[str]:
        query_lower = query_string.lower()
        if " headshot" in query_lower:
            player_name = query_lower.replace(" headshot", "").strip()
            return ' '.join(word.capitalize() for word in player_name.split())
        return None

    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        logger.info(f"[{self.name}] Starting static asset retrieval pipeline.")
        ctx.session.state["current_static_assets_list"] = []

        current_recap = ctx.session.state.get("current_recap")
        if not current_recap:
            logger.warning(f"[{self.name}] 'current_recap' missing. Skipping static asset pipeline.")
            yield Event(
                author=self.name,
                content=types.Content(role="model", parts=[types.Part(text="Static asset pipeline skipped: no recap.")])
            )
            return

        logger.info(f"[{self.name}] Running internal EntityExtractorForAssets...")
        async for event in self.entity_extractor.run_async(ctx):
            yield event

        logger.info(f"[{self.name}] Running internal StaticAssetQueryGenerator...")
        async for event in self.static_asset_query_generator.run_async(ctx):
            yield event

        queries_json_str = ctx.session.state.get("static_asset_search_queries_json", "[]")
        cleaned_queries_json_str = _clean_json_string_from_llm(queries_json_str)
        search_queries = []
        try:
            parsed_queries = json.loads(cleaned_queries_json_str)
            if isinstance(parsed_queries, list):
                search_queries = [str(q) for q in parsed_queries if isinstance(q, str)]
        except json.JSONDecodeError as e:
            logger.error(f"[{self.name}] Failed to parse static_asset_search_queries_json: {e}. String: '{cleaned_queries_json_str}'")

        player_lookup_dict_json_str = ctx.session.state.get("player_lookup_dict_json", "{}")
        player_lookup_dict = {}
        player_name_to_id_map_lower = {}
        try:
            cleaned_lookup_str = _clean_json_string_from_llm(player_lookup_dict_json_str, default_if_empty="{}")
            player_lookup_dict = json.loads(cleaned_lookup_str)
            if not isinstance(player_lookup_dict, dict):
                logger.warning(f"[{self.name}] player_lookup_dict_json parsed into non-dict: {type(player_lookup_dict)}. Using empty map.")
                player_lookup_dict = {}
            player_name_to_id_map_lower = {
                str(name).lower(): str(id_val) 
                for name, id_val in player_lookup_dict.items() 
                if name and id_val is not None
            }
            if player_lookup_dict:
                 logger.info(f"[{self.name}] Parsed player_lookup_dict_json. Lowercase map size: {len(player_name_to_id_map_lower)}")
            else:
                logger.info(f"[{self.name}] player_lookup_dict_json was empty after parsing.")
        except json.JSONDecodeError:
            logger.error(f"[{self.name}] Failed to parse player_lookup_dict_json: '{player_lookup_dict_json_str}'. Using empty map.")
        
        found_assets_list = []
        logger.info(f"[{self.name}] Processing {len(search_queries)} asset search queries.")

        for original_query_string in search_queries:
            logger.debug(f"[{self.name}] Processing query: '{original_query_string}'")
            team_name_for_logo = self._extract_team_name_from_query(original_query_string)
            player_name_for_headshot = self._extract_player_name_from_query(original_query_string)

            if team_name_for_logo:
                logger.info(f"[{self.name}] Identified LOGO query for team: '{team_name_for_logo}'")
                ctx.session.state["team_name_for_logo_search"] = team_name_for_logo
                async for event in self.logo_searcher_llm.run_async(ctx):
                    yield event
                
                logo_tool_output_str = ctx.session.state.get("logo_search_result_json", "[]") # Default to "[]" for list
                cleaned_logo_tool_output_str = _clean_json_string_from_llm(logo_tool_output_str, default_if_empty='[]')
                logger.debug(f"[{self.name}] Raw output from LogoSearcherLlm for '{team_name_for_logo}': '{cleaned_logo_tool_output_str[:300]}...'")

                try:
                    # Assuming hyper-strict prompt makes LogoSearcherLlm output the tool's direct JSON payload (a list string)
                    logo_results_list = json.loads(cleaned_logo_tool_output_str)
                    
                    if logo_results_list and isinstance(logo_results_list, list) and \
                       len(logo_results_list) > 0 and isinstance(logo_results_list[0], dict) and \
                       logo_results_list[0].get("image_uri"):
                        asset_info = logo_results_list[0] # Take the first result
                        asset_info["search_term_origin"] = original_query_string
                        found_assets_list.append(asset_info)
                        logger.info(f"[{self.name}] Successfully parsed and added logo: {asset_info.get('image_uri')} for {team_name_for_logo}")
                    elif isinstance(logo_results_list, dict) and logo_results_list.get("error"): # Check if tool itself returned an error JSON
                        logger.warning(f"[{self.name}] LogoSearcherLlm tool reported an error for '{team_name_for_logo}': {logo_results_list.get('error')}")
                    else:
                        logger.warning(f"[{self.name}] Parsed logo data for '{team_name_for_logo}', but no valid image_uri or not expected list structure. Data: {logo_results_list}")
                except json.JSONDecodeError as e:
                    logger.error(f"[{self.name}] Failed to parse JSON from LogoSearcherLlm's output for '{team_name_for_logo}'. Error: {e}. String was: '{cleaned_logo_tool_output_str}'")
                except Exception as e_generic:
                    logger.error(f"[{self.name}] Unexpected error processing logo result for '{team_name_for_logo}': {e_generic}", exc_info=True)

            elif player_name_for_headshot:
                logger.info(f"[{self.name}] Identified HEADSHOT query for player: '{player_name_for_headshot}'")
                player_id = player_name_to_id_map_lower.get(player_name_for_headshot.lower())
                original_player_name_casing = player_lookup_dict.get(str(player_id), player_name_for_headshot) if player_id else player_name_for_headshot


                if player_id:
                    logger.info(f"[{self.name}] Found Player ID '{player_id}' for '{player_name_for_headshot}'. Original casing for tool: '{original_player_name_casing}'")
                    ctx.session.state["player_id_for_headshot_search"] = str(player_id)
                    ctx.session.state["player_name_for_headshot_log"] = original_player_name_casing
                    async for event in self.headshot_retriever_llm.run_async(ctx):
                        yield event

                    headshot_tool_output_str = ctx.session.state.get("headshot_uri_result_json", "{}") # Default to "{}" for dict
                    cleaned_headshot_tool_output_str = _clean_json_string_from_llm(headshot_tool_output_str, default_if_empty='{}')
                    logger.debug(f"[{self.name}] Raw output from HeadshotRetrieverLlm for '{original_player_name_casing}': '{cleaned_headshot_tool_output_str[:300]}...'")
                    
                    try:
                        # Assuming hyper-strict prompt makes HeadshotRetrieverLlm output the tool's direct JSON object string
                        headshot_result_dict = json.loads(cleaned_headshot_tool_output_str)
                        
                        if headshot_result_dict and isinstance(headshot_result_dict, dict) and headshot_result_dict.get("image_uri"):
                            asset_info = headshot_result_dict
                            asset_info["search_term_origin"] = original_query_string
                            found_assets_list.append(asset_info)
                            logger.info(f"[{self.name}] Successfully parsed and added headshot: {asset_info.get('image_uri')} for {original_player_name_casing}")
                        elif isinstance(headshot_result_dict, dict) and headshot_result_dict.get("error"):
                             logger.warning(f"[{self.name}] HeadshotRetrieverLlm tool reported an error for '{original_player_name_casing}': {headshot_result_dict.get('error')}")
                        elif isinstance(headshot_result_dict, dict) and not headshot_result_dict.get("image_uri"): # Valid JSON dict, but no URI
                            logger.warning(f"[{self.name}] Headshot found for '{original_player_name_casing}' (ID: {player_id}) but no 'image_uri' in response: {headshot_result_dict}")
                        else:
                            logger.warning(f"[{self.name}] Parsed headshot data for '{original_player_name_casing}', but not expected dict structure or missing URI. Data: {headshot_result_dict}")
                    except json.JSONDecodeError as e:
                        logger.error(f"[{self.name}] Failed to parse JSON from HeadshotRetrieverLlm's output for '{original_player_name_casing}'. Error: {e}. String was: '{cleaned_headshot_tool_output_str}'")
                    except Exception as e_generic_hs:
                        logger.error(f"[{self.name}] Unexpected error processing headshot result for '{original_player_name_casing}': {e_generic_hs}", exc_info=True)
                else:
                    logger.warning(f"[{self.name}] Player ID NOT FOUND for headshot query: '{player_name_for_headshot}' (searched as '{player_name_for_headshot.lower()}')")
            else:
                logger.warning(f"[{self.name}] Unrecognized asset query format, skipping: '{original_query_string}'")
        
        ctx.session.state["current_static_assets_list"] = found_assets_list
        thought_text = f"Static asset pipeline complete. Found {len(found_assets_list)} static assets."
        logger.info(f"[{self.name}] {thought_text}")
        if found_assets_list: # Log details only if assets were found
            try:
                logger.info(f"[{self.name}] Details of found static assets: {json.dumps(found_assets_list, indent=2)}")
            except TypeError: # In case something non-serializable slipped in (shouldn't happen with this logic)
                logger.error(f"[{self.name}] Could not serialize found_assets_list for logging.")


        yield Event(
             author=self.name,
             content=types.Content(
                 role="model",
                 parts=[types.Part(text=thought_text)]
            )
        )


    # IterativeImageGenerationAgent will now orchestrate these and the subsequent critique/refinement.
    # It needs to handle the asynchronous nature of the LR tool completion.
    # For simplicity, we'll assume the "agent client" handles polling, and this agent
    # picks up the result from a known state key once the FunctionResponse is processed.
    # A more robust CustomAgent would be needed for intricate internal polling and state management
    # if the agent client wasn't handling it.

    # visual_critic_agent and new_visual_prompts_from_critique_agent remain the same LlmAgents.

    # REVISED IterativeImageGenerationAgent (now a SequentialAgent for clarity of flow)
    # The loop logic for refinement would be handled by the overarching GameRecapAgentV2 or a LoopAgent within.
    # For now, let's assume ONE round of generation + critique for this refactoring.
    # The original IterativeImageGenerationAgent had a loop; this needs careful thought.
    # Let's make it a sequence for one pass, and the "agent client" handles the LR part.
    # The loop from the original IterativeImageGenerationAgent is complex with an external polling mechanism.
    # We will simplify: Generate -> Client Polls -> Result in state -> Critique -> New Prompts.
    # The *retry/refinement* loop part of IterativeImageGenerationAgent is harder with LR tools
    # if the agent itself isn't the one doing the polling and reacting.

    # Let's define a new Custom Agent for the iterative image part.
    class ADKNativeIterativeImageGenerationAgent(BaseAgent): # Inherit from BaseAgent for custom logic
        model_config = {"arbitrary_types_allowed": True}
        prompt_generator: LlmAgent
        lr_tool_caller: LlmAgent # Calls the LongRunningFunctionTool
        visual_critic: LlmAgent
        new_prompts_creator: LlmAgent
        max_refinement_loops: int

        # Key where the LlmAgent (lr_tool_caller) will eventually store the *final result*
        # of the long-running image generation after the client injects the FunctionResponse.
        LR_IMAGE_RESULT_STATE_KEY: str = "generated_visual_assets_uris_json_from_lr"


        def __init__(self, name: str,
                     prompt_generator: LlmAgent,
                     lr_tool_caller: LlmAgent,
                     visual_critic: LlmAgent,
                     new_prompts_creator: LlmAgent,
                     max_visual_refinement_loops: int = 1):
            super().__init__(
                name=name,
                prompt_generator=prompt_generator,
                lr_tool_caller=lr_tool_caller,
                visual_critic=visual_critic,
                new_prompts_creator=new_prompts_creator,
                max_visual_refinement_loops=max_visual_refinement_loops,
                # Sub-agents for ADK framework visibility.
                # The actual calls will be managed in _run_async_impl
                sub_agents=[prompt_generator, lr_tool_caller, visual_critic, new_prompts_creator]
            )

        @override
        async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
            logger.info(f"[{self.name}] Starting ADK-native iterative image generation workflow.")
            ctx.session.state["all_generated_image_assets_details"] = [] # Final list for this agent
            
            # Ensure game_pk is available for the tool caller
            if "game_pk" not in ctx.session.state: # Assuming game_pk is set by an earlier agent
                logger.warning(f"[{self.name}] 'game_pk' missing. Image generation might have issues with naming.")
                ctx.session.state["game_pk"] = "unknown_game_for_image_lr"


            # Initial prompt generation
            logger.info(f"[{self.name}] Running GeneratedVisualPromptsForLR...")
            async for event in self.prompt_generator.run_async(ctx): yield event
            
            current_prompts_json_for_tool = ctx.session.state.get("visual_generation_prompts_json", "[]")
            if not current_prompts_json_for_tool or current_prompts_json_for_tool == "[]":
                logger.info(f"[{self.name}] No visual prompts generated. Skipping image generation loop.")
                yield Event(author=self.name, content=types.Content(role="model", parts=[types.Part(text="No image prompts, skipping generation.")]))
                return

            all_generated_assets_details_for_this_agent = []

            for i in range(self.max_visual_refinement_loops + 1): # +1 for initial generation
                iteration_label = f"Refinement Iteration {i}" if i > 0 else "Initial Generation"
                logger.info(f"[{self.name}] {iteration_label} - Loop {i+1}/{self.max_visual_refinement_loops + 1}")

                # State key for prompts for the LR tool caller might need to be distinct per iteration if prompts change
                # For now, visual_generation_prompts_json is updated by new_prompts_creator for next iter
                ctx.session.state["visual_generation_prompts_json_for_lr_tool_caller"] = current_prompts_json_for_tool

                logger.info(f"[{self.name}] Running ImageGenerationLRToolCaller for prompts: {current_prompts_json_for_tool[:100]}...")
                # This call will make the LlmAgent call the LongRunningFunctionTool.
                # The LlmAgent will then PAUSE, and this ADKNativeIterativeImageGenerationAgent will also pause.
                # The "agent client" (e.g. in main.py) must now poll for the task_id given by the LRFT's initial response
                # and inject the FunctionResponse.
                async for event in self.lr_tool_caller.run_async(ctx):
                    yield event
                    # We expect lr_tool_caller to set "image_generation_lr_task_submission_details"
                    # with {"status": "pending_agent_client_action", "task_id": ..., "tool_name": ...}
                    # The actual result (URIs) will only be available *after* the client polls and injects a FunctionResponse.
                    # That FunctionResponse will be processed by the lr_tool_caller LlmAgent in a *subsequent* agent run pass,
                    # and it will then write its final output (the URIs) to its output_key.

                # === This is the tricky part with LongRunningFunctionTool and agent-internal loops ===
                # The agent is PAUSED HERE until the client injects the FunctionResponse.
                # When the agent RESUMES, the lr_tool_caller would have processed the FunctionResponse.
                # The result of the LR tool (the image URIs) will be in lr_tool_caller.output_key.
                # We need a way for this _run_async_impl to "wait" or be re-entered after the LRFT completes.

                # Let's assume that when this agent resumes, the result is in `image_generation_lr_task_submission_details`
                # NO, the LlmAgent `lr_tool_caller`'s output_key will hold the final FunctionResponse content.
                # Let `lr_tool_caller.output_key` be `lr_image_tool_final_output`.
                
                # This agent's `_run_async_impl` will complete its current iteration after yielding events from `lr_tool_caller`.
                # The `GameRecapAgentV2` will resume, and if it's a `SequentialAgent`, it will proceed to the next agent *or*
                # if the `ADKNativeIterativeImageGenerationAgent` is the one looping, its `_run_async_impl` needs to be
                # re-entrant or designed as a state machine.

                # For strict adherence to LRFT doc: Agent calls LRFT -> Agent PAUSES -> Client polls & injects FunctionResponse -> Agent RESUMES.
                # The loop for refinement must happen *after* the image generation is fully resolved.

                # Let's adjust the design:
                # 1. ImagePromptGeneratorAgent
                # 2. ImageGenerationLRToolCallerAgent (calls LRFT, pauses)
                # --- External Client Polling and FunctionResponse Injection ---
                # (Agent resumes, ImageGenerationLRToolCallerAgent now has the image URIs in its output_key)
                # 3. ImageResultCollectorAgent (A simple BaseAgent to take output from ImageGenerationLRToolCallerAgent.output_key
                #    and place it into `assets_for_critique_json` and `all_generated_image_assets_details`).
                # 4. VisualCriticAgent
                # 5. NewVisualPromptsCreatorAgent (updates `visual_generation_prompts_json` for the next loop iteration)
                # These 5 steps would be inside a LoopAgent.

                # This means `ADKNativeIterativeImageGenerationAgent` should be a `LoopAgent` itself,
                # or orchestrated by one. The `_run_async_impl` cannot directly manage a loop
                # that depends on an externally completed LRFT within the same continuous flow.

                # Simplified: For THIS iteration of the refactor, the iterative loop for IMAGE generation
                # will be removed. It will be: GenPrompts -> CallLRFT -> (Client Polls & Responds) -> Agent gets URI -> Critique (once).
                # The `max_refinement_loops` will effectively be 0 for the LR part.
                # The critique part will run once after the first batch of images.

                # When this ADKNativeIterativeImageGenerationAgent's _run_async_impl is called AGAIN by the runner
                # (after the LRFT's FunctionResponse has been processed and lr_tool_caller completed its turn),
                # then `ctx.session.state[self.lr_tool_caller.output_key]` will hold the URIs.
                
                # This requires the agent client to manage re-running this agent or the parent.
                # This is where CustomAgent's `_run_async_impl` shines for complex stateful flows.

                # Let's assume for now:
                # - `lr_tool_caller` is run. It makes the tool call. This agent's `_run_async_impl` yields and effectively ends this "turn".
                # - The agent client polls, gets image URIs for the `task_id` that `lr_tool_caller` initiated.
                # - The client injects a `FunctionResponse`.
                # - The `Runner` processes this, eventually `lr_tool_caller` processes the `FunctionResponse` and writes its `output_key`.
                # - The *next time* `ADKNativeIterativeImageGenerationAgent` (or its parent) is run, the image URIs are available.

                # This means `ADKNativeIterativeImageGenerationAgent` cannot have its own simple for-loop for refinement
                # if the image generation step within that loop is an LRFT handled by the client.
                # It should rather be structured as a state machine or a sequence of agents where the "client action" is a break point.

                # Let's simplify `ADKNativeIterativeImageGenerationAgent` for this refactor to do ONE pass:
                # GenPrompts -> CallLRFT (and PAUSE).
                # The critique and new prompts part will be handled by a *separate subsequent agent*
                # once the LRFT result is available in state.

                # So, this agent becomes simpler:
                # (In _run_async_impl of a new orchestrator for image generation)
                # 1. Run self.prompt_generator
                # 2. Run self.lr_tool_caller
                # (This agent's job for this "turn" is done. It has initiated the LR task)
                # The result processing and critique will be handled by other agents in the main sequence
                # after the client has done its part.

                # For the purpose of returning *something* for the `all_generated_image_assets_details`
                # it would be populated *after* the LR tool has completed and its results are in state.
                # This implies that the agent that populates `all_generated_image_assets_details`
                # runs *after* the LR tool is fully resolved.

                # Let's define what this agent DOES:
                # It generates prompts, initiates the LR image gen.
                # The actual processing of results and critique happens later in the main sequence.
                # So, this agent should probably be split or simplified.

                # Path for this refactor:
                # 1. `GeneratedVisualPromptsForLR` (LlmAgent) -> outputs `visual_generation_prompts_json`
                # 2. `ImageGenerationLRToolCallerAgent` (LlmAgent) -> calls LRFT, outputs `image_generation_lr_task_submission_details`
                #    (The runner loop now polls and injects FunctionResponse for the tool call made by ImageGenerationLRToolCallerAgent)
                #    (When ImageGenerationLRToolCallerAgent resumes, its output_key gets the actual image URIs as a JSON string)
                # 3. `ImageResultProcessingAgent` (New BaseAgent or LlmAgent)
                #    - Input: reads `ImageGenerationLRToolCallerAgent.output_key` (which contains JSON string of URIs)
                #    - Parses it.
                #    - Populates `all_generated_image_assets_details` in session state.
                #    - Populates `assets_for_critique_json` in session state.
                # 4. `VisualCriticAgent` (LlmAgent) - uses `assets_for_critique_json`
                # 5. `NewVisualPromptsCreatorAgent` (LlmAgent) - uses `visual_critique_text` (for a potential next iteration if a LoopAgent orchestrates this)

                # `ADKNativeIterativeImageGenerationAgent` can be a `SequentialAgent` of these new components.
                # The "iterative" part would mean this whole sequence is looped by a parent `LoopAgent`.

                # For now, this agent is a ONE-PASS sequence:
                current_prompts_list_for_iter = []
                try:
                    parsed_list = json.loads(current_prompts_json_for_tool)
                    if isinstance(parsed_list, list) and parsed_list:
                        current_prompts_list_for_iter = [str(p) for p in parsed_list if isinstance(p, str) and p]
                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(f"[{self.name}] Invalid JSON/content for prompts. Error: {e}. JSON: '{current_prompts_json_for_tool}'.")
                    yield Event(author=self.name, content=types.Content(role="model", parts=[types.Part(text="Error with image prompts.")]))
                    return
                
                if not current_prompts_list_for_iter:
                    logger.info(f"[{self.name}] No valid prompts to process for image generation.")
                    yield Event(author=self.name, content=types.Content(role="model", parts=[types.Part(text="No valid prompts for image generation.")]))
                    return

                # This agent's main job is to kick off the LRFT call via its sub_agent
                # It doesn't directly process the final image URIs in this same run.
                # That happens after the client polling.
                # The `all_generated_image_assets_details` will be populated by a *subsequent* agent in the sequence.
                thought = f"[{self.name}] {iteration_label}: Image generation initiated via LRToolCaller. Client will poll."
                logger.info(thought)
                yield Event(author=self.name, content=types.Content(role="model", parts=[types.Part(text=thought)]))

                # The actual image URI processing and adding to `all_generated_assets_details_for_this_agent`
                # will happen in a *later* agent in the sequence, AFTER the LR Tool is resolved by the client.
                # So, this agent doesn't directly populate `all_generated_image_assets_details`.
                # A different agent will read `ImageGenerationLRToolCallerAgent.output_key` later.
                
                # If this agent is supposed to handle the full loop (including waiting for LRFT), it MUST be a CustomAgent
                # with its own internal polling logic that interacts with the external client's polling. This is too complex for now.

                # Let's assume this simplified agent just runs the first two steps for ONE set of prompts.
                # The critique/refinement part will be handled by `AssetValidationAndRetryAgent` or a similar orchestrator
                # that runs AFTER the LR tools are resolved.

            # This simplified agent does not loop internally for LRFTs.
            # It initiates one batch of image generation.
            # The `all_generated_image_assets_details` will be populated by another agent.
            logger.info(f"[{self.name}] Image generation initiation phase complete.")
            # No further processing of URIs or critique within THIS agent for now.
            # That logic needs to be shifted to an agent that runs AFTER client polling.




################################################################################
# --- TextToSpeechAgent (Copied - Assumed Correct) ---
################################################################################
class TextToSpeechAgent(BaseAgent):
    model_config = {"arbitrary_types_allowed": True}
    dialogue_to_speech_llm_agent: DirectToolCallerBaseAgent

    def __init__(self, name: str, dialogue_to_speech_llm_agent: DirectToolCallerBaseAgent):
        super().__init__(name=name, dialogue_to_speech_llm_agent=dialogue_to_speech_llm_agent, sub_agents=[dialogue_to_speech_llm_agent])

    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        logger.info(f"[{self.name}] Starting Text-to-Speech generation.")
        ctx.session.state["generated_dialogue_audio_uri"] = None
        if not ctx.session.state.get("current_recap"):
            logger.warning(f"[{self.name}] 'current_recap' missing. Skipping TTS.")
            yield Event(author=self.name, type="agent_thought", content=types.Content(role="model", parts=[types.Part(text="TTS generation skipped: no recap script.")]))
            return
        async for event in self.dialogue_to_speech_llm_agent.run_async(ctx): yield event
        cleaned_audio_details_json = _clean_json_string_from_llm(ctx.session.state.get("generated_dialogue_audio_details_json", "{}"), default_if_empty='{}')
        try:
            audio_details = json.loads(cleaned_audio_details_json)
            if isinstance(audio_details, dict):
                if audio_details.get("error"): logger.error(f"[{self.name}] TTS LlmAgent error: {audio_details['error']}")
                else:
                    potential_uri = audio_details.get("uri") or audio_details.get("audio_uri")
                    if potential_uri and isinstance(potential_uri, str) and potential_uri.startswith("gs://"):
                        ctx.session.state["generated_dialogue_audio_uri"] = potential_uri
        except json.JSONDecodeError as e:
            logger.error(f"[{self.name}] Failed to parse JSON from TTS LlmAgent. Error: {e}.")
        logger.info(f"[{self.name}] TTS complete. Audio URI: {ctx.session.state['generated_dialogue_audio_uri']}")
        thought_text = f"TTS generation complete. Audio URI: {ctx.session.state.get('generated_dialogue_audio_uri')}"
        logger.info(f"[{self.name}] {thought_text}")
        thought_text = f"TTS generation complete. Audio URI: {ctx.session.state.get('generated_dialogue_audio_uri')}"
        logger.info(f"[{self.name}] {thought_text}")
        yield Event(author=self.name, content=types.Content(role="model", parts=[types.Part(text=thought_text)]))


################################################################################
# --- SpeechToTimestampsAgent (Copied - Assumed Correct) ---
################################################################################
class SpeechToTimestampsAgent(BaseAgent):
    model_config = {"arbitrary_types_allowed": True}
    audio_to_timestamps_llm_agent: DirectToolCallerBaseAgent

    def __init__(self, name: str, audio_to_timestamps_llm_agent: DirectToolCallerBaseAgent):
        super().__init__(name=name, audio_to_timestamps_llm_agent=audio_to_timestamps_llm_agent, sub_agents=[audio_to_timestamps_llm_agent])

    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        logger.info(f"[{self.name}] Starting Speech-to-Timestamps generation.")
        ctx.session.state["word_timestamps_list"] = []
        audio_uri_for_stt = ctx.session.state.get("generated_dialogue_audio_uri")
        if not audio_uri_for_stt:
            logger.warning(f"[{self.name}] 'generated_dialogue_audio_uri' missing. Skipping STT.")
            yield Event(author=self.name, type="agent_thought", content=types.Content(role="model", parts=[types.Part(text="Timestamp generation skipped: no audio URI.")]))
            return
        async for event in self.audio_to_timestamps_llm_agent.run_async(ctx): yield event
        # The LlmAgent "audio_to_timestamps_llm_for_audio" is instructed to output the direct JSON string from the tool.
        raw_timestamps_json_from_llm = ctx.session.state.get("word_timestamps_json", "[]")
        cleaned_timestamps_json = _clean_json_string_from_llm(raw_timestamps_json_from_llm, default_if_empty='[]')
        try:
            timestamps_data = json.loads(cleaned_timestamps_json)
            if isinstance(timestamps_data, list):
                ctx.session.state["word_timestamps_list"] = timestamps_data
            elif isinstance(timestamps_data, dict) and timestamps_data.get("error"):
                 logger.error(f"[{self.name}] STT LlmAgent error: {timestamps_data['error']}")
            else: # If it's not a list or an error dict from the tool, log it.
                logger.warning(f"[{self.name}] STT LlmAgent returned unexpected JSON: {cleaned_timestamps_json}")
        except json.JSONDecodeError as e:
            logger.error(f"[{self.name}] Failed to parse JSON from STT LlmAgent. Error: {e}. String was: '{cleaned_timestamps_json}'")
        logger.info(f"[{self.name}] STT complete. Timestamps: {len(ctx.session.state.get('word_timestamps_list',[]))}")

        thought_text = f"STT generation complete. Timestamps: {len(ctx.session.state.get('word_timestamps_list',[]))}"
        logger.info(f"[{self.name}] {thought_text}")
        yield Event(author=self.name, content=types.Content(role="model", parts=[types.Part(text=thought_text)]))


################################################################################
# --- NEW GameRecapAgentV2 (Parallelized) ---
# This agent will be a SequentialAgent.
# It will first run script generation, then a ParallelAgent for assets,
# then the validator.
################################################################################
# Note: We will define this within `create_agent_with_preloaded_tools`
# as it's composed of other agents defined there.





################################################################################
# ---  AssetAggregatorAgent Definition ---
################################################################################
class AssetAggregatorAgent(BaseAgent):
    model_config = {"arbitrary_types_allowed": True, "extra": "ignore"}

    # Use ClassVar to tell Pydantic these are not instance fields
    STATE_KEYS_TO_AGGREGATE: ClassVar[Dict[str, str]] = {
        "static_assets": "current_static_assets_list",
        "generated_images": "all_generated_image_assets_details",
        "generated_videos": "final_video_assets_list",
        "generated_graphs": "generated_graph_assets_details_json",
        "dialogue_audio": "generated_dialogue_audio_uri"
    }
    OUTPUT_KEY_ALL_VISUALS: ClassVar[str] = "final_aggregated_visual_assets_list"
    OUTPUT_KEY_AUDIO: ClassVar[str] = "final_aggregated_audio_asset"

    def __init__(self, name: str, **kwargs):
        super().__init__(name=name, **kwargs)
        # No need to handle the ClassVar attributes in __init__ for Pydantic purposes

    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        logger.info(f"[{self.name}] Starting asset aggregation.")

        aggregated_visuals = []
        aggregated_audio = None
        final_recap_text = ctx.session.state.get("current_recap", "No recap text found.")

        # Access ClassVar using self or AssetAggregatorAgent
        static_assets_key = self.STATE_KEYS_TO_AGGREGATE["static_assets"]
        generated_images_key = self.STATE_KEYS_TO_AGGREGATE["generated_images"]
        generated_videos_key = self.STATE_KEYS_TO_AGGREGATE["generated_videos"]
        generated_graphs_key = self.STATE_KEYS_TO_AGGREGATE["generated_graphs"]
        dialogue_audio_key = self.STATE_KEYS_TO_AGGREGATE["dialogue_audio"]

        # 1. Aggregate Static Assets
        static_assets_raw = ctx.session.state.get(static_assets_key, [])
        if isinstance(static_assets_raw, list):
            for asset in static_assets_raw:
                if isinstance(asset, dict) and asset.get("image_uri"):
                    aggregated_visuals.append({
                        "type": asset.get("type", "static_image"),
                        "uri": asset.get("image_uri"),
                        "source_query": asset.get("search_term_origin", "N/A"),
                        "details": asset
                    })
            logger.info(f"[{self.name}] Aggregated {len(static_assets_raw)} raw static assets from key '{static_assets_key}'.")
        else:
            logger.warning(f"[{self.name}] Static assets in state ('{static_assets_key}') was not a list: {type(static_assets_raw)}")

        # 2. Aggregate Generated Images
        generated_images_raw = ctx.session.state.get(generated_images_key, [])
        if isinstance(generated_images_raw, list):
            for asset in generated_images_raw:
                if isinstance(asset, dict) and asset.get("image_uri"):
                    aggregated_visuals.append({
                        "type": asset.get("type", "generated_image"),
                        "uri": asset.get("image_uri"),
                        "prompt_origin": asset.get("prompt_origin", "N/A"),
                        "iteration": asset.get("iteration", 0),
                        "details": asset
                    })
            logger.info(f"[{self.name}] Aggregated {len(generated_images_raw)} raw generated images from key '{generated_images_key}'.")
        else:
            logger.warning(f"[{self.name}] Generated images in state ('{generated_images_key}') was not a list: {type(generated_images_raw)}")

        # 3. Aggregate Generated Videos
        generated_videos_raw = ctx.session.state.get(generated_videos_key, [])
        if isinstance(generated_videos_raw, list):
            for asset in generated_videos_raw:
                if isinstance(asset, dict) and asset.get("video_uri"):
                    aggregated_visuals.append({
                        "type": asset.get("type", "generated_video"),
                        "uri": asset.get("video_uri"),
                        "source_prompt_index": asset.get("source_prompt_index", -1),
                        "details": asset
                    })
            logger.info(f"[{self.name}] Aggregated {len(generated_videos_raw)} raw generated videos from key '{generated_videos_key}'.")
        else:
             logger.warning(f"[{self.name}] Generated videos in state ('{generated_videos_key}') was not a list: {type(generated_videos_raw)}")

        # 4. Aggregate Generated Graphs
        generated_graphs_json_str = ctx.session.state.get(generated_graphs_key, "[]")
        generated_graphs_raw = [] # Default
        try:
            cleaned_graphs_json_str = _clean_json_string_from_llm(generated_graphs_json_str, default_if_empty="[]")
            parsed_graphs = json.loads(cleaned_graphs_json_str)
            if isinstance(parsed_graphs, list):
                generated_graphs_raw = parsed_graphs # Keep the list of dicts
                for asset in generated_graphs_raw:
                    if isinstance(asset, dict) and asset.get("status") == "success" and asset.get("graph_image_uri"):
                        aggregated_visuals.append({
                            "type": asset.get("type", "generated_graph"),
                            "uri": asset.get("graph_image_uri"),
                            "title": asset.get("title", "N/A"),
                            "graph_id": asset.get("graph_id", "N/A"),
                            "details": asset
                        })
                logger.info(f"[{self.name}] Processed {len(generated_graphs_raw)} raw generated graph entries from key '{generated_graphs_key}'.")
            else:
                logger.warning(f"[{self.name}] Parsed generated graphs ('{generated_graphs_key}') was not a list: {type(parsed_graphs)}")
        except json.JSONDecodeError:
            logger.error(f"[{self.name}] Failed to parse state key '{generated_graphs_key}': {generated_graphs_json_str}")

        # 5. Aggregate Audio Asset
        audio_uri_raw = ctx.session.state.get(dialogue_audio_key)
        if isinstance(audio_uri_raw, str) and audio_uri_raw.startswith("gs://"):
            aggregated_audio = {
                "type": "dialogue_audio",
                "uri": audio_uri_raw,
                "word_timestamps_count": len(ctx.session.state.get("word_timestamps_list", [])),
            }
            logger.info(f"[{self.name}] Aggregated audio URI: {audio_uri_raw} from key '{dialogue_audio_key}'.")
        else:
            logger.warning(f"[{self.name}] Dialogue audio URI in state ('{dialogue_audio_key}') was not a valid string URI: {audio_uri_raw}")

        ctx.session.state[self.OUTPUT_KEY_ALL_VISUALS] = aggregated_visuals
        ctx.session.state[self.OUTPUT_KEY_AUDIO] = aggregated_audio

        aggregation_summary = (
            f"Asset Aggregation Complete.\n"
            f"Total Visual Assets (images, videos, graphs): {len(aggregated_visuals)}\n"
            f"Audio Asset URI: {aggregated_audio.get('uri') if aggregated_audio else 'None'}\n"
            f"Recap Text (first 100 chars for log): {final_recap_text[:100]}..."
        )
        # For the final event to the user, just provide the recap.
        # The aggregated assets are in the state for downstream systems.
        logger.info(f"[{self.name}] Aggregation summary: Visuals: {len(aggregated_visuals)}, Audio: {'Yes' if aggregated_audio else 'No'}")
        
        # Decide what this agent's final "user-facing" output should be.
        # It could be the recap text, or a summary of assets, or both.
        # If it's just an aggregator, its "thought" might be enough, and the previous agent's output is the final one.
        # However, since it's the last in the main sequence, its output becomes the sequence's output.
        # Let's output the original recap text, as the aggregation is primarily for internal state.
        yield Event(
            author=self.name,
            content=types.Content(role="model", parts=[types.Part(text=final_recap_text)]) # Output the original recap
        )

# --- Tool Collection (Copied - Assumed Correct) ---
async def _collect_tools_stack(
    server_config_dict: AllServerConfigs,
) -> Tuple[Dict[str, Any], contextlib.AsyncExitStack]:
    all_tools: Dict[str, Any] = {}
    exit_stack = contextlib.AsyncExitStack()
    stack_needs_closing = False
    try:
        if not hasattr(server_config_dict, "configs") or not isinstance(server_config_dict.configs, dict):
            return {}, exit_stack
        for key, server_params in server_config_dict.configs.items():
            individual_exit_stack: Optional[contextlib.AsyncExitStack] = None
            try:
                tools, individual_exit_stack = await MCPToolset.from_server(connection_params=server_params)
                if individual_exit_stack:
                    await exit_stack.enter_async_context(individual_exit_stack)
                    stack_needs_closing = True
                if tools: all_tools[key] = tools
            except Exception as e:
                logging.error(f"Failed to connect/get tools for {key}: {e}", exc_info=True)
        expected_tool_keys = [
            "weather", "bnb", "ct", "mlb", "web_search", "bq_search",
            "visual_assets", "static_retriever_mcp", "image_embedding_mcp",
            "video_clip_generator_mcp", "audio_processing_mcp", "game_plotter_tool",
        ]
        for k in expected_tool_keys:
            if k not in all_tools: all_tools[k] = []
        return all_tools, exit_stack
    except Exception as e:
        if stack_needs_closing: await exit_stack.aclose()
        raise


# --- Agent Creation (Refactored for Parallel Game Recap) ---
async def create_agent_with_preloaded_tools(
    loaded_mcp_tools: Dict[str, Any],
) -> LlmAgent: # Root agent is still an LlmAgent

    # --- Standard Assistant Agents (Copied) ---
    booking_tools = loaded_mcp_tools.get("bnb", []) + loaded_mcp_tools.get("weather", [])
    ct_tools = loaded_mcp_tools.get("ct", [])
    mlb_tools = loaded_mcp_tools.get("mlb", [])
    # web_search_tools = loaded_mcp_tools.get("web_search", []) # Used below
    # bq_search_tools = loaded_mcp_tools.get("bq_search", []) # Used below

    booking_agent = LlmAgent(model=MODEL_ID, name="booking_assistant", instruction="Use tools for booking/weather. Markdown. agent_exit if unknown.", tools=booking_tools)
    cocktail_agent = LlmAgent(model=MODEL_ID, name="cocktail_assistant", instruction="Use tools for cocktails. Markdown. agent_exit if unknown.", tools=ct_tools)
    mlb_assistant = LlmAgent(model=MODEL_ID, name="mlb_assistant", instruction="MLB Stats assistant. Use tools for stats. Ask for IDs. Markdown. agent_exit if unknown.", tools=mlb_tools)

    # --- Sub-Agents for GameRecapAgentV2 (Copied LlmAgent definitions) ---
    # These are the "leaf" LlmAgents performing specific tasks.
    # Their definitions (instructions, tools, output_keys) are assumed to be correct from your provided code.

    query_setup_agent_instance = QuerySetupAgent(name="QuerySetup")

    initial_recap_generator_agent = LlmAgent(
        name="InitialRecapGenerator", model=GEMINI_PRO_MODEL_ID,
        instruction="""
You are an expert sports journalist for an MLB game recap in a **two-host dialogue script format**.
Session State: `game_pk` (may be initially unset), `user_query`, `parsed_game_pk_from_query` (Optional), `pre_game_context_notes` (Optional), `past_critiques_feedback` (Optional).

Your Multi-Step Process:

1.  **Game Identification (Critical):**
    *   If `game_pk` is not already set in session state: Analyze `user_query`.
    *   If the query implies a "latest" game (e.g., "Brewers last game"), use `mlb.get_team_schedule` (e.g., `days_range=-7`) to find the most recent *final* game. Extract its `game_pk`.
    *   If a `game_pk` is found or provided: Your response text MUST include a line formatted EXACTLY AS: `INTERNAL_GAME_PK_FOUND: [game_pk_value_here]`. For example: `INTERNAL_GAME_PK_FOUND: 717527`. This line should be separate, typically at the beginning or end of your textual output.
    *   If no specific game can be confidently identified (e.g., query is too vague, no recent final games found), your response text MUST include the line: `INTERNAL_GAME_PK_FOUND: NONE`.
    *   After this line, you can optionally add a human-readable announcement like "Okay, I've identified the game..." if a PK was found.

2.  **Gather Core Game Data (if a `game_pk_value_here` was identified, not "NONE"):**
    *   Use the identified `game_pk_value_here` for all subsequent tool calls needing a game_pk.
    *   Call `mlb.get_live_game_score`.
    *   Call `mlb.get_game_play_by_play_summary` (e.g., 15-20 plays).
    *   Call `mlb_stats.get_game_boxscore_and_details`.
    *   Parse the data: game status, final score, winning/losing pitchers, key performers, etc.
    *   **If Game Status is "Scheduled" or not "Final"**: Your recap dialogue MUST state that a full recap is not yet available as the game is not final. Then use `agent_exit` by including that function call.

3.  **Synthesize Initial Dialogue Script (Only if Game is "Final" and a `game_pk_value_here` was identified):**
    *   **Dialogue Format:** Conversation between two hosts. **Strict Alternation.** **NO Speaker Labels.**
    *   **Storytelling First:** Primary goal is to tell the story of the game. Identify a "story of the game."
    *   **Pitching Narrative:** Discuss winner/loser.
    *   **Offensive Highlights & Progression:** Cover scoring, key players.
    *   **Integrate Context:** Use `pre_game_context_notes`, `past_critiques_feedback`.
    *   **Language:** Vivid, active, natural.

Output ONLY the generated recap text (which must include the `INTERNAL_GAME_PK_FOUND:` line). Do not add conversational fluff like "Here is the recap..." unless it's part of the dialogue script itself.
If you use `agent_exit`, ensure the `INTERNAL_GAME_PK_FOUND:` line still precedes it if a PK was determined.
        """,
        tools=[tool for tool_set_name in ["bq_search", "mlb", "web_search"] for tool in loaded_mcp_tools.get(tool_set_name, [])],
        output_key="current_recap",
    )

    # NEW: Instantiate the GameStateSetterAgent
    game_state_setter_agent_instance = GameStateSetterAgent(name="GameStateSetter")

    # This LlmAgent extracts team names from the recap for the PlayerIdPopulator
    entity_extractor_for_player_id_agent = LlmAgent(
        name="EntityExtractorForPlayerIdPopulation",
        model=GEMINI_FLASH_MODEL_ID,
        instruction="""Read the dialogue script from session state 'current_recap'.
Identify all unique full MLB team names (e.g., "Milwaukee Brewers", "Pittsburgh Pirates") explicitly mentioned.
Output a JSON object: {"teams": ["Team Name 1", "Team Name 2"]}.
If no teams are clearly identified, output {"teams": []}.
Your output MUST be ONLY this JSON string.
        """,
        output_key="extracted_teams_for_roster_json" # Distinct key
    )

    player_id_populator_agent_instance = PlayerIdPopulatorAgent(name="PlayerIdPopulator", mlb_tools=mlb_tools)
    # PlayerIdPopulatorAgent's _run_async_impl will internally create an LlmAgent
    # to call mlb.get_team_roster using teams from "extracted_teams_for_roster_json" (it will need to be adapted to read this key)
    # and will set "player_lookup_dict_json".

    recap_critic_agent = LlmAgent(
        name="RecapCritic", model=GEMINI_PRO_MODEL_ID, # Using PRO for better critique
        instruction="""
You are a sharp MLB analyst/producer critiquing a dialogue script.
Session: `current_recap`, `game_pk`, `user_query`.
Review `current_recap` for:
- **Dialogue Flow & Engagement:** Natural? Engaging back-and-forth? Distinct voices? Real discussion?
- **Accuracy & Completeness (in dialogue):** Scores, key actions, sequence detailed?
- **Narrative (of dialogue):** Compelling story? Arc? Tension/excitement? Engaging language?
- **Journalistic Style (of dialogue):** Professional sports podcast?
- **Info Gaps & Enrichment (in dialogue):** What's missing? Opportunities for more stats/context?
- **Clarity & Flow (of dialogue):** Easy to follow? Clear lines?
- **Data Usage (by hosts):** Stats used effectively?
If excellent: "The recap is excellent." ONLY. Else, **specific, bulleted feedback** with examples for dialogue.
        """,
        output_key="current_critique",
    )
    critique_processor_agent = LlmAgent(
        name="CritiqueProcessor", model=GEMINI_PRO_MODEL_ID,
        instruction="""
Research assistant processing `current_critique`. Session: `game_pk`, `user_query`.
1.  **Store Critique:** Call `bq_search.store_new_critique` with `critique_text`={current_critique}, `task_text`="recap for game_pk {game_pk} from '{user_query}'", `game_pk_str` (string of game_pk or empty), `revision_number_str` (from state or empty). Let result be `critique_storage_status_json`.
2.  **Gen Web Queries:** From `current_critique`, create 1-3 Tavily queries for info gaps (missing details, trends, context).
3.  **Web Search:** For each query (max 3), call `web_search.perform_web_search` (`max_results=1` or 2). Collect results into `web_search_findings_list`.
4.  **RAG Search:** Call `bq_search.search_rag_documents` with `query_text`={current_critique}, `game_pk_str`, `top_n=2`. Result `rag_findings_json_list`.
5.  **Final JSON Output (ONLY this string):**
    ```json
    {{
      "critique_storage_status": "{{\"status\":\"success\", ...}}", // from step 1
      "web_search_queries_generated": ["query1", ...], // from step 2
      "web_search_findings": ["Tavily: Finding 1", ...], // from step 3
      "rag_findings": ["RAG: Doc content 1", ...], // from step 4, parsed
      "overall_status_message": "Critique processed. Searches performed."
    }}
    ```
    Use empty lists if no data for a key.
        """,
        tools=[tool for tool_set_name in ["bq_search", "web_search"] for tool in loaded_mcp_tools.get(tool_set_name, [])],
        output_key="critique_processor_results_json",
    )
    recap_reviser_agent = LlmAgent(
        name="RecapReviser", model=GEMINI_PRO_MODEL_ID,
        instruction="""
Expert sports story editor revising a game recap **dialogue script**.
Session: `current_recap`, `current_critique`, `critique_processor_results_json` (JSON string with `web_search_findings`, `rag_findings`), `user_query`, `past_critiques_feedback` (Optional), core game data.
1.  **Parse Research:** Extract `web_search_findings`, `rag_findings` from `critique_processor_results_json`.
2.  **Address Critique (Dialogue Format):** Revise `current_recap` (dialogue) for *every* actionable point in `current_critique`.
    *   **Integrate Research:** Weave web/RAG findings into dialogue (e.g., one host presents, other reacts).
    *   **Enhance Story/Flow:** Elevate language. Natural back-and-forth.
    *   **Refine Narrative Arc:** Clear lead, key moments, effective conclusion in dialogue.
    *   **Contextualize:** Hosts discuss stats and significance.
3.  **Stylistic Guidance:** Apply `past_critiques_feedback` for conversational sports commentary.
4.  **Fulfill Request:** Ensure revised dialogue addresses `user_query`.
5.  **Clarity/Conciseness:** Clear host lines, good flow.
6.  **Maintain Dialogue Format:** **Strict Alternation. NO Speaker Labels.**
Output ONLY the revised dialogue script. No intros/outros.
        """,
        output_key="current_recap",
    )
    grammar_check_agent = LlmAgent(
        name="RecapGrammarCheck", model=GEMINI_FLASH_MODEL_ID,
        instruction="Grammar/style checker. Session: `current_recap`. Review for errors, awkward phrasing, impact. Output JSON list of suggestions, or `[]` or `[\"Grammar and style are good.\"]` if excellent.",
        output_key="grammar_suggestions",
    )
    tone_check_agent = LlmAgent(
        name="RecapToneCheck", model=GEMINI_FLASH_MODEL_ID,
        instruction="Tone analyzer. Session: `current_recap`. Analyze from winner/neutral perspective. Output ONLY: 'positive', 'negative', or 'neutral'.",
        output_key="tone_check_result",
    )
    generated_visual_prompts_agent = LlmAgent( # Was: GeneratedVisualPrompts
        name="GeneratedVisualPromptsForLR", # New name for clarity
        model=GEMINI_FLASH_MODEL_ID,
        instruction="""
Assistant director analyzing 'current_recap' for Imagen 3 shots (filters names).
Identify 3-5 key moments/scenes.
**Imagen Rules:** NO Player/Team Names. Generic uniforms ("player in white home uniform", "batter in colored away jersey", "MLB player's uniform").
**Prompts:** Actions (HR, DP, K) -> 1-2 prompts. Descriptive moments (stadium) -> 1 prompt. Emphasize action, emotion, setting, generic uniforms.
Output ONLY JSON list of 3-5 prompt strings (e.g., "[\\"Prompt 1\\", ...]"). "[]" if no moments.
        """,
        output_key="visual_generation_prompts_json", # This key holds the prompts List[str] for the LR tool
    )

    # 2. Agent to call the LongRunningFunctionTool for Image Generation
    image_generation_lr_tool_caller_agent = LlmAgent(
        name="ImageGenerationLRToolCaller",
        model=GEMINI_FLASH_MODEL_ID, # Simple model to just make a tool call
        instruction="""
You are a dispatcher. Your task is to initiate an image generation job.
Read the list of image prompts from session state key 'visual_generation_prompts_json'.
Read the game PK from session state key 'game_pk'.
You MUST call the 'initiate_image_generation' tool with these prompts and the game_pk_str.
Your entire job is to make this single tool call. Do not respond with text.
The tool will return a task ID and status. This will be your output.
        """,
        tools=[long_running_image_tool], # Pass the ADK LongRunningFunctionTool
        output_key="final_image_uris_from_lr_tool_json_string" # Will contain {"status": "pending_agent_client_action", "task_id": ...}
    )

    # Create the sequential agent for one pass of image generation initiation and result collection
    image_result_processor_agent = LlmAgent( # Can be an LlmAgent if it just needs to parse and set state
        name="ImageResultProcessorFromLR",
        model=GEMINI_FLASH_MODEL_ID, # Or even a BaseAgent if no LLM needed
        instruction="""
You are a data processor. You will receive a JSON string in session state key '{lr_image_tool_final_output}'.
This string contains a list of image URIs or an error from an image generation tool.
Parse this JSON string.
If it's a list of URIs, create a new list of asset detail objects: `[{"image_uri": "uri1", "type": "generated_image_lr", "prompt_origin":"unknown_for_now"}, ...]`
Store this list of asset detail objects in session state key 'all_generated_image_assets_details'.
Also, prepare a JSON string for 'assets_for_critique_json' using these URIs and dummy prompts (e.g., `[{"prompt_origin": "Prompt for URI1", "image_uri": "uri1"}, ...]`).
If the input JSON string indicates an error, or is not a list, set 'all_generated_image_assets_details' to `[]` and 'assets_for_critique_json' to `"[]"`.
Output a brief status message like "Processed N image URIs." or "Image generation failed."
        """,
        # This agent needs to know the output_key of `image_generation_lr_tool_caller_agent`
        # Let's assume `image_generation_lr_tool_caller_agent.output_key` is "lr_image_tool_final_output"
        # This is injected via placeholder in instruction.
        # This LlmAgent approach is a bit forced for simple data manipulation. A BaseAgent would be cleaner.
        output_key="image_result_processing_status"
    )
# final_image_uris_from_lr_tool_json_string

    iterative_image_generation_pipeline_components = [
        generated_visual_prompts_agent, # Generates prompts into `visual_generation_prompts_json`
        image_generation_lr_tool_caller_agent, # Calls LRFT, result (after client poll) in `image_generation_lr_task_submission_details` (actually its output_key)
        # At this point, the main runner loop polls and injects FunctionResponse.
        # The `image_generation_lr_tool_caller_agent` then completes and writes to its output_key.
        # We need to use THAT output_key as input for the next agent.
        # Let's say image_generation_lr_tool_caller_agent.output_key = "final_image_uris_from_lr_tool_json_string"
    ]
    # The critique part will be added after this sequence, consuming `all_generated_image_assets_details`
    # which `image_result_processor_agent` would set.

    # For now, `iterative_image_generation_agent_instance` is simplified to just initiate.
    # The full iterative loop with LR tools requires more advanced orchestration.
    # Let's define the agent that *will* be used in the ParallelAssetPipelinesAgent
    # This will be a sequence that does: GenPrompts -> CallLRFT
    # The result processing and critique must happen *after* the ParallelAgent completes its branches.
    image_generation_initiation_sequence = SequentialAgent(
        name="ImageGenerationInitiationSequence",
        sub_agents=[
            generated_visual_prompts_agent,
            image_generation_lr_tool_caller_agent
        ],
        description="Generates prompts and initiates long-running image generation."
    )
    # The output of `image_generation_lr_tool_caller_agent` (task details / final URIs after polling)
    # will be used by later agents in the main GameRecapAgentV2 sequence.  IterativeImageGeneration

    # == VIDEO GENERATION REVISED ==
    # 1. Agent to generate prompts (same as before)
    veo_prompt_generator_agent = LlmAgent( # Was: VeoPromptGenerator
        name="VeoPromptGeneratorForLR", # New name
        model=GEMINI_PRO_MODEL_ID,
        instruction="""
Creative video director for MLB Veo clips (5-8s). Prompts MUST be safe.
Session: 'current_recap', 'visual_critique_text' (Optional from image phase), 'all_image_assets_list' (Optional), 'all_video_assets_list' (Optional).
1.  **Review 'current_recap'**: ID 2-3 key moments for short clips (pitch/swing, dive, slide, reaction, fan celebration). Avoid complex sequences.
2.  **Consider Existing Visuals**: Review 'all_image_assets_list', 'all_video_assets_list'. Avoid duplication unless video adds unique dynamism. Complement, don't repeat.
3.  **Veo Prompts (1-2, max 3 if distinct & safe):**
    *   **Safety First:** Adhere to content guidelines. Factual descriptions. AVOID aggressive/violent words (e.g., "batter hits long fly ball" not "crushes"). Focus athleticism.
    *   **Clarity:** One primary subject/action.
    *   **Conciseness:** Scene for 5-8s.
    *   **Dynamic but Safe Language:** Motion, camera work ("slow-motion of...", "dynamic low-angle...").
    *   **Naming Rules (Strict):** NO player/team names. Generic: "MLB player", "home team batter".
Output ONLY JSON list of 1-2 (max 3) Veo prompt strings. `"[]"` if no suitable moments. Example: `["Slow-motion of baseball hitting bat.", "Dynamic shot of player sliding."]`
        """,
        output_key="veo_generation_prompts_json", # This key holds the prompts List[str] for the LR tool
    )

    # 2. Agent to call the LongRunningFunctionTool for Video Generation
    video_generation_lr_tool_caller_agent = LlmAgent(
        name="VideoGenerationLRToolCaller",
        model=GEMINI_FLASH_MODEL_ID,
        instruction="""
You are a dispatcher. Your task is to initiate a video generation job.
Read the list of video prompts from session state key 'veo_generation_prompts_json'.
Read the game PK from session state key 'game_pk'.
You MUST call the 'initiate_video_generation' tool with these prompts and the game_pk_str.
Your entire job is to make this single tool call. Do not respond with text.
The tool will return a task ID and status. This will be your output.
        """,
        tools=[long_running_video_tool],
        output_key="final_video_uris_from_lr_tool_json_string"
    )

    video_generation_initiation_sequence = SequentialAgent(
        name="VideoGenerationInitiationSequence",
        sub_agents=[
            veo_prompt_generator_agent,
            video_generation_lr_tool_caller_agent
        ],
        description="Generates prompts and initiates long-running video generation."
    )


    # Asset-related LlmAgents (copied)
    entity_extractor_agent = LlmAgent(
        name="EntityExtractorForAssets", model=GEMINI_FLASH_MODEL_ID,
        instruction="""Read 'current_recap'. Identify unique full player names and MLB team names. Output JSON: {"players": ["Player Full Name", ...], "teams": ["Full Team Name", ...]}. Empty lists if none. ONLY JSON string.""",
        output_key="extracted_entities_json",
    )
    static_asset_query_generator_agent = LlmAgent(
        name="StaticAssetQueryGenerator", model=GEMINI_FLASH_MODEL_ID,
        instruction="""Parse 'extracted_entities_json'. For each team, query "[Team Name] logo". For each player, query "[Player Name] headshot". Output JSON list of these query strings. "[]" if no entities. ONLY JSON string.""",
        output_key="static_asset_search_queries_json",
    )
    logo_searcher_llm_agent = LlmAgent(
        name="LogoSearcherLlm", model=GEMINI_PRO_MODEL_ID,
        instruction="""
Your ONLY GOAL is to call the `image_embedding_mcp.search_similar_images_by_text` tool EXACTLY ONCE and output its RAW JSON string result.

STATE INPUT: `session.state.team_name_for_logo_search`

STEPS:
1. Check if `session.state.team_name_for_logo_search` is present and not empty.
2. IF EMPTY OR MISSING: Your output MUST be the exact JSON string: `"[]"`. DO NOTHING ELSE.
3. IF PRESENT: Call `image_embedding_mcp.search_similar_images_by_text` ONCE with parameters:
    - `query_text`: value from `session.state.team_name_for_logo_search`
    - `top_k`: 1
    - `filter_image_type`: "logo"
4. The tool will return a JSON string.
5. Your FINAL and ONLY output MUST be this exact JSON string from the tool.
DO NOT add "Okay". DO NOT add "Here it is". DO NOT GREET. DO NOT SUMMARIZE. DO NOT ASK QUESTIONS. Output ONLY the JSON string.
""",
        tools=loaded_mcp_tools_global.get("image_embedding_mcp", []),
        output_key="logo_search_result_json",
    )
    headshot_retriever_llm_agent = LlmAgent(
        name="HeadshotRetrieverLlm", model=GEMINI_PRO_MODEL_ID,
        instruction="""
You are a robot with a single, precise function.
1.  You will be given `player_id_for_headshot_search` and `player_name_for_headshot_log` in session state.
2.  You MUST call the tool `static_retriever_mcp.get_headshot_uri_if_exists` EXACTLY ONCE.
3.  Use these exact parameters for the tool call:
    - `player_id_str`: {session.state.player_id_for_headshot_search}
    - `player_name_for_log`: {session.state.player_name_for_headshot_log}
4.  The tool will return a JSON string. This JSON string is your ONLY output.
5.  Your entire response for this step MUST be the direct, verbatim JSON string that is returned by the tool.
6.  DO NOT add any other text, explanation, formatting, markdown, or conversational remarks.
7.  If `session.state.player_id_for_headshot_search` is missing or empty, your output should be the JSON string "{}" (an empty object). DO NOT call the tool in this case.
""",
        tools=loaded_mcp_tools_global.get("static_retriever_mcp", []),
        output_key="headshot_uri_result_json",
    )


    visual_generator_mcp_caller_agent = LlmAgent(
        name="VisualGeneratorMCPCaller", model=GEMINI_FLASH_MODEL_ID, # Changed from PRO
        instruction="""
Image gen coordinator.
1.  Inputs: `session.state.visual_generation_prompts_json_for_tool` (JSON string list), `session.state.game_pk_str_for_tool`. If prompts invalid/empty, output `"[]"` ONLY, no tool call.
2.  Tool: Call `visual_assets.generate_images_from_prompts` with `prompts_json`=input prompts string, `game_pk_str`=input game_pk string.
3.  Output (CRITICAL): Tool responds like `{"name":"...", "response":{"result":{"content":[{"text":"[\\"gs://uri1\\",...]"}]}}}`. Extract inner `"text"` value.
    Final output MUST be ONLY this extracted JSON string (e.g., `"[\\"gs://uri1\\", ...]"`) or tool's error JSON (e.g., `"{\\"error\\":...}"`). No other text.
        """,
        tools=loaded_mcp_tools_global.get("visual_assets", []),
        output_key="generated_visual_assets_uris_json",
    )
    visual_critic_agent = LlmAgent(
        name="VisualCritic", model=GEMINI_FLASH_MODEL_ID, # Changed from PRO
        instruction="""
Visual producer reviewing generated images (Imagen 3 limits names).
Session: 'current_recap', 'assets_for_critique_json' (`[{"prompt_origin": "...", "image_uri": "gs://..." or null}]`), 'prompts_used_for_critique_json'.
Critique:
1.  **Relevance:** Image vs. prompt vs. 'current_recap'? Action gaps for successful gens?
2.  **Failures:** Prompts with null image_uri?
3.  **Quality (successful gens):** Clear? Convey action/mood from prompt?
4.  **Suggestions (Generator-Safe):** If needed, suggest NEW prompts for missing actions/composition. NO player/team names.
If ALL good or no initial prompts: "Visuals look sufficient." ONLY. Else, bullet-point feedback & SPECIFIC, SAFE prompt suggestions for next round.
        """,
        output_key="visual_critique_text",
    )
    new_visual_prompts_from_critique_agent = LlmAgent(
        name="NewVisualPromptsFromCritique", model=GEMINI_FLASH_MODEL_ID,
        instruction="""
Assistant director refining visual plans from 'visual_critique_text'.
**Strict Limits:** NO Player/Team Names, Generic Uniforms.
Task: If critique "Visuals look sufficient." or empty, output `"[]"`. Else, identify concepts. Generate JSON list of 2-4 NEW, concise, specific, SAFE prompt strings. Translate specific names from critique to generic. Focus action, setting, emotion.
Output ONLY JSON list string. Example: "[\\"Dynamic shot of fielders turning double play...\\"]"
        """,
        output_key="new_visual_generation_prompts_json",
    )

    video_generator_mcp_caller_agent = LlmAgent(
        name="VideoGeneratorMCPCaller", model=GEMINI_PRO_MODEL_ID,
        instruction="""
You are a highly specialized, non-conversational robot. Your SOLE AND ONLY function is to manage a SINGLE tool call to `video_clip_generator_mcp.generate_video_clips_from_prompts` and then IMMEDIATELY output its raw JSON string result. You have NO OTHER CAPABILITIES OR TASKS.

Follow these steps EXACTLY and in this order. DEVIATION IS NOT PERMITTED:

1.  **Retrieve Inputs from Session State:**
    *   Get the JSON string from session state key `veo_generation_prompts_json_for_tool`.
    *   Get the string from session state key `game_pk_str_for_tool`.

2.  **Input Validation & Preparation:**
    *   Attempt to parse the string from `veo_generation_prompts_json_for_tool` into a list of prompt strings. Let this be `parsed_prompt_list`.
    *   Let the string from `game_pk_str_for_tool` be `game_pk_value`.

3.  **Conditional Tool Call (Perform ONCE or NOT AT ALL):**
    *   IF `parsed_prompt_list` is empty, or if it's not a list after parsing, or if `veo_generation_prompts_json_for_tool` was initially missing/empty:
        *   Your ONLY output for this entire interaction MUST be the exact JSON string: `"[]"`
        *   DO NOT call any tool. YOUR TASK ENDS HERE.
    *   ELSE (if `parsed_prompt_list` is valid and contains prompts):
        *   You MUST call the tool `video_clip_generator_mcp.generate_video_clips_from_prompts` EXACTLY ONE TIME.
        *   Use the `parsed_prompt_list` (the actual Python list of strings) as the value for the tool's `prompts` parameter.
        *   Use `game_pk_value` as the value for the tool's `game_pk_str` parameter.

4.  **Output Generation (CRITICAL - Precise Echo):**
    *   The `video_clip_generator_mcp.generate_video_clips_from_prompts` tool will return a JSON string to you (this string will represent either a list of GCS URIs for generated videos or an error object from the tool).
    *   Your entire and final output for this entire interaction MUST be ONLY this direct, verbatim JSON string that was returned by the tool from that single call.
    *   DO NOT analyze, modify, reformat, or add any text, markdown, explanations, or conversational remarks before or after this JSON string.
    *   DO NOT call the tool `video_clip_generator_mcp.generate_video_clips_from_prompts` more than once under any circumstances.
    *   Your task is COMPLETE once you have outputted the tool's raw JSON string response from the single call.

Example of your required output if tool is called and succeeds: `["gs://video1.mp4", "gs://video2.mp4"]` (as a JSON string)
Example of your required output if tool is called and tool itself returns an error: `{"error": "tool specific error details"}` (as a JSON string)
Example of your required output if initial prompts are invalid/empty (Step 3.IF condition): `"[]"` (as a JSON string)
        """,
        tools=loaded_mcp_tools_global.get("video_clip_generator_mcp", []),
        output_key="generated_video_clips_uris_json",
    )
    dialogue_to_speech_llm_for_audio = LlmAgent(
        name="DialogueToSpeechLlmForGameRecap", model=GEMINI_PRO_MODEL_ID, # Using PRO for potentially complex scripts
        instruction="""
Audio synthesis robot. SOLE RESPONSIBILITY: generate speech via tool.
IMMEDIATELY call `audio_processing_mcp.synthesize_multi_speaker_speech`.
Params: `script`={session.state.current_recap}, `game_pk_str`={session.state.game_pk}.
Output MUST BE tool's direct, verbatim JSON string (e.g., `{"audio_uri": "gs://..."}` or `{"error": "..."}`). No other text.
        """,
        tools=loaded_mcp_tools_global.get("audio_processing_mcp", []),
        output_key="generated_dialogue_audio_details_json",
    )
    audio_to_timestamps_llm_for_audio = LlmAgent(
        name="AudioToTimestampsLlmForGameRecap", model=GEMINI_PRO_MODEL_ID,
        instruction="""
Audio transcription robot. ONLY manage tool call & return specific string.
1.  Input: `session.state.generated_dialogue_audio_uri`. If missing/invalid (not gs://), output `{"error": "Invalid audio GCS URI."}` ONLY, no tool call.
2.  Tool: Call `audio_processing_mcp.get_word_timestamps_from_audio` with `audio_gcs_uri`=input URI.
3.  Output (CRITICAL): Tool responds `{"name":"...", "response":{"result":{"content":[{"text":"[{\"word\": ...}]"}]}}}`. Extract inner `"text"` value.
    Final output MUST be ONLY this extracted JSON string (e.g., `"[{\"word\": ...}]"`) or tool's error JSON. No other text.
        """,
        tools=loaded_mcp_tools_global.get("audio_processing_mcp", []),
        output_key="word_timestamps_json",
    )

    # --- Define Tools for Graph Generation ---
    game_plotter_tools = loaded_mcp_tools.get("game_plotter_tool", [])
    if not game_plotter_tools:
        logger.warning("GamePlotterTool toolset not found in loaded_mcp_tools. Graph generation may fail.")

    # --- Agents for Graph Generation ---
    graphable_metric_selector_agent = LlmAgent(
        name="GraphableMetricSelector",
        model=GEMINI_PRO_MODEL_ID,
        instruction="""You are a sports data analyst and storyteller.
Your goal is to identify 1-2 compelling data visualizations (graphs) that would enhance an MLB game recap.
INPUTS available in session state:
- `game_pk`: The unique identifier for the game.
- `current_recap`: The textual dialogue of the game recap.
- `raw_game_stats_json` (Optional, if a previous agent fetched detailed stats like play-by-play or boxscore in JSON format).

TASK:
1.  Analyze the `current_recap` to understand the game's narrative, key moments, and standout player performances.
2.  If `raw_game_stats_json` is available, analyze it. If not, or if more specific data is needed, you can use `mlb` tools (like `mlb.get_game_boxscore_and_details`, `mlb.get_game_play_by_play_summary` for the given `game_pk`) to fetch necessary data to inform your choices.
3.  Based on the narrative and data, decide on 1 or (at most) 2 distinct metrics that would be insightful to visualize. Examples:
    *   Win probability progression throughout the game.
    *   Pitch velocity/type distribution for a key pitcher.
    *   Team scoring by inning.
    *   Comparison of key offensive player stats (e.g., Hits, HRs, RBIs for top 2-3 players).
    *   Run differential over innings.
4.  For each chosen visualization, specify the following in a JSON object:
    *   `graph_id`: A short, unique, descriptive ID (e.g., "win_probability_chart", "pitcher_X_velo_breakdown", "team_scoring_timeline").
    *   `chart_type`: The preferred chart type (e.g., "line", "bar", "scatter", "pie").
    *   `title`: A clear and concise title for the graph.
    *   `x_axis_label`: Label for the X-axis.
    *   `y_axis_label`: Label for the Y-axis.
    *   `data_to_plot_description`: A natural language description of the exact data needed for this plot. This description MUST be detailed enough for another agent to fetch or prepare the data. For example:
        *   "Home team's win probability after each completed half-inning."
        *   "For pitcher [Pitcher's Name from recap/stats], plot pitch types on X-axis and their counts on Y-axis for this game."
        *   "Bar chart showing runs scored by each team in each inning."
    *   `source_data_tool_hint` (Optional but helpful): If you know specific `mlb` tool calls (and parameters) that would provide the raw data for this plot, suggest them. E.g., "Use mlb.get_game_play_by_play_summary and extract win probability fields."
5.  Your final output MUST be a JSON string representing a list of these graph instruction objects.
    Example: `[{"graph_id": "win_prob", "chart_type": "line", ..., "data_to_plot_description": "Home team win probability per inning"}, {"graph_id": "player_OBP", ...}]`
If no compelling or feasible graphs can be identified from the available data or recap context, output an empty JSON list: `[]`.
Output ONLY the JSON string.
        """,
        tools=mlb_tools, # Access to MLB tools to understand data context
        output_key="graph_plotting_instructions_json"
    )

    graph_generator_agent = LlmAgent(
        name="GraphGenerator",
        model=GEMINI_PRO_MODEL_ID, # Needs to understand instructions and prepare data for the tool
        instruction="""You are a data preparation and graph generation coordinator.
INPUTS available in session state:
- `game_pk`: The unique identifier for the game.
- `graph_plotting_instructions_json`: A JSON string list of graph tasks, where each task has `graph_id`, `chart_type`, `title`, `x_axis_label`, `y_axis_label`, `data_to_plot_description`, and optionally `source_data_tool_hint`.
- `raw_game_stats_json` (Optional, containing previously fetched game data).

TASK:
For each graph task object in `graph_plotting_instructions_json`:
1.  Read the `data_to_plot_description` and `source_data_tool_hint`.
2.  Fetch the precise data needed for the plot using `mlb` tools and the `game_pk`. You might need to process data from `mlb.get_game_boxscore_and_details`, `mlb.get_game_play_by_play_summary`, etc. Transform this data into simple lists suitable for plotting (e.g., a list for x-values, a list for y-values, or a list of series for multi-series plots).
3. Prepare arguments for `game_plotter_tool.generate_graph`.
   The `game_pk_str` should be the game PK as a string.
   The `x_data_json` argument MUST be a STRING containing a JSON array (e.g., `x_data_json = "[1, 2, 3]"`).
   The `y_data_json` argument MUST be a STRING containing a JSON array (e.g., `y_data_json = "[10, 15, 20]"`).
   The `data_series_json` argument MUST be a STRING containing a JSON array of series objects (e.g., `data_series_json = "[{\"label\": \"A\", \"y_values\": [1,2]}]"`).
   The `options_json` argument MUST be a STRING containing a JSON object (e.g., `options_json = "{\"color\":\"blue\"}"`).
   If any of these data components are not applicable or empty, pass the string "[]" for list types or "{}" for object types.
   DO NOT pass Python lists or dicts directly as values for these *_json arguments; they must be JSON formatted strings.
4.  Call `game_plotter_tool.generate_graph` with these prepared arguments.
5.  Collect all successfully generated graph details (which include `graph_image_uri`) from the tool responses.
Your final output MUST be a JSON string list of successfully generated graph asset detail objects. Each object should be the JSON returned by the tool, e.g., `{"status": "success", "graph_id": "...", "graph_image_uri": "gs://...", "title": "...", "type": "generated_graph"}`.
If a graph generation fails for a specific task, log it mentally but try to generate others.
If `graph_plotting_instructions_json` is empty or no graphs are successfully generated, output an empty JSON list: `[]`.
Output ONLY the JSON string list.
        """,
        tools=[*mlb_tools, *game_plotter_tools], # Needs MLB tools and the new plotter tool
        output_key="generated_graph_assets_details_json" # This state key will hold the list of graph asset URIs
    )


   # 3. For Logo Search (replaces LogoSearcherLlm LlmAgent)
    direct_logo_searcher_agent = DirectToolCallerBaseAgent(
        name="DirectLogoSearcher",
        tool_name_to_call="image_embedding_mcp.search_similar_images_by_text", # From your image_embedding_server.py
        input_state_keys={ # Tool args for search_similar_images_by_text
            "query_text": "team_name_for_logo_search", # Set by StaticAssetPipelineAgent
            "top_k": "1", # Will pass None, tool should have default or handle. Or set fixed value.
            "filter_image_type": "" # Same as above.
            # For fixed values, we can also hardcode them here if the LlmAgent was doing that.
            # Or, the LlmAgent that SETS team_name_for_logo_search can also set fixed args for the tool.
            # Let's assume fixed values are better set by the tool or the agent preparing the call.
            # For now, we will pass them as None and let the tool handle defaults or error.
            # A better way: these fixed args should be part of the agent's __init__ or set in state.
            # Let's modify the DirectToolCaller to allow fixed args.
        },
        output_state_key="logo_search_result_json", # StaticAssetPipelineAgent reads this
        default_output_on_error='"[]"' # Expects a JSON list string
        # We'll need to enhance DirectToolCallerBaseAgent for fixed args or ensure they are in state.
    )

    # 4. For Headshot Retrieval (replaces HeadshotRetrieverLlm LlmAgent)
    direct_headshot_retriever_agent = DirectToolCallerBaseAgent(
        name="DirectHeadshotRetriever",
        tool_name_to_call="static_retriever_mcp.get_headshot_uri_if_exists", # From your static_asset_retriever_mcp_server.py
        input_state_keys={
            "player_id_str": "player_id_for_headshot_search", # Set by StaticAssetPipelineAgent
            "player_name_for_log": "player_name_for_headshot_log" # Set by StaticAssetPipelineAgent
        },
        output_state_key="headshot_uri_result_json", # StaticAssetPipelineAgent reads this
        default_output_on_error='{}' # Expects a JSON object string
    )

    # 1. For Visual Generation (replaces VisualGeneratorMCPCaller LlmAgent)
    direct_visual_generator_agent = DirectToolCallerBaseAgent(
        name="DirectVisualGenerator",
        tool_name_to_call="visual_assets.generate_images_from_prompts", # From your visual_asset_server.py
        input_state_keys={ # Tool arg name : session state key
            "prompts": "visual_generation_prompts_json_for_tool", # This state key is set by GeneratedVisualPrompts agent
            "game_pk_str": "game_pk_str_for_tool"      # This state key is set by IterativeImageGenerationAgent
        },
        output_state_key="generated_visual_assets_uris_json", # IterativeImageGenerationAgent reads this
        default_output_on_error='"[]"' # Expects JSON string of a list
    )

# GeneratedVisualPromptsForLR

    # 2. For Video Generation (replaces VideoGeneratorMCPCaller LlmAgent)
    direct_video_generator_agent = DirectToolCallerBaseAgent(
        name="DirectVideoGenerator",
        tool_name_to_call="video_clip_generator_mcp.generate_video_clips_from_prompts", # From your video_clip_server.py
        input_state_keys={
            "prompts": "veo_generation_prompts_json_for_tool", # Set by VeoPromptGenerator agent
            "game_pk_str": "game_pk_str_for_tool"      # Set by VideoPipelineAgent before this call
        },
        output_state_key="generated_video_clips_uris_json", # VideoPipelineAgent reads this
        default_output_on_error='"[]"'
    )
    # 5. For TTS (replaces DialogueToSpeechLlmForGameRecap LlmAgent)
    direct_tts_agent = DirectToolCallerBaseAgent(
        name="DirectTTSGenerator",
        tool_name_to_call="audio_processing_mcp.synthesize_multi_speaker_speech",
        input_state_keys={
            "script": "current_recap", # Assuming current_recap is the final script text
            "game_pk_str": "game_pk"   # Assuming game_pk state holds the string ID or number
        },
        output_state_key="generated_dialogue_audio_details_json", # TextToSpeechAgent (BaseAgent wrapper) reads this
        default_output_on_error='{}'
    )

    # 6. For STT (replaces AudioToTimestampsLlmForGameRecap LlmAgent)
    direct_stt_agent = DirectToolCallerBaseAgent(
        name="DirectSTTGenerator",
        tool_name_to_call="audio_processing_mcp.get_word_timestamps_from_audio",
        input_state_keys={
            "audio_gcs_uri": "generated_dialogue_audio_uri" # Set by TextToSpeechPipeline/DirectTTSGenerator
        },
        output_state_key="word_timestamps_json", # SpeechToTimestampsAgent (BaseAgent wrapper) reads this
        default_output_on_error='"[]"'
    )

    graph_generation_pipeline_agent_instance = SequentialAgent(
        name="GraphGenerationPipeline",
        sub_agents=[
            graphable_metric_selector_agent,
            graph_generator_agent
        ],
        description="Selects appropriate metrics and generates graph images for the game."
    )

    # StaticAssetPipelineAgent, TextToSpeechAgent, SpeechToTimestampsAgent, GraphGenerationPipeline remain the same
    # as they don't use DirectToolCallerBaseAgent internally for their primary function or are already LlmAgents/custom.
    # The `logo_searcher_llm` and `headshot_retriever_llm` inside `StaticAssetPipelineAgent`
    # were LlmAgents, not DirectToolCallerBaseAgent.
    # Checking: `logo_searcher_llm_agent` and `headshot_retriever_llm_agent` were indeed LlmAgents in your original code.
    # If they were `DirectToolCallerBaseAgent`, they'd need similar refactoring, but they were not.

    # The agents `direct_logo_searcher_agent`, `direct_headshot_retriever_agent`,
    # `direct_visual_generator_agent`, `direct_video_generator_agent`, `direct_tts_agent`, `direct_stt_agent`
    # were all instances of `DirectToolCallerBaseAgent` and are being replaced or their callers are.
    # `visual_generator_mcp_caller` was used by `IterativeImageGenerationAgent` -> replaced by `image_generation_lr_tool_caller_agent`.
    # `video_generator_mcp_caller` was used by `VideoPipelineAgent` -> replaced by `video_generation_lr_tool_caller_agent`.

    # Update `StaticAssetPipelineAgent` to ensure its sub-agents are LlmAgents as previously defined,
    # not the `DirectToolCallerBaseAgent` versions that were placeholders for MCP replacement.
    # The `logo_searcher_llm_agent` and `headshot_retriever_llm_agent` should be the LlmAgent versions provided earlier.

    # --- Instantiate Custom Phase Agents (BaseAgent subclasses) ---
    static_asset_pipeline_agent_instance = StaticAssetPipelineAgent(
        name="StaticAssetPipeline",
        entity_extractor=entity_extractor_agent, # This is an LlmAgent
        static_asset_query_generator=static_asset_query_generator_agent, # This is an LlmAgent
        logo_searcher_llm=direct_logo_searcher_agent, # This MUST be the LlmAgent version
        headshot_retriever_llm=direct_headshot_retriever_agent # This MUST be the LlmAgent version
    )


    text_to_speech_agent_instance = TextToSpeechAgent( # This is your BaseAgent wrapper
        name="TextToSpeechPipeline",
        dialogue_to_speech_llm_agent=direct_tts_agent 
    )
    speech_to_timestamps_agent_instance = SpeechToTimestampsAgent( # This is your BaseAgent wrapper
        name="SpeechToTimestampsPipeline",
        audio_to_timestamps_llm_agent=direct_stt_agent #
    )

    # --- Define Workflow Agents for GameRecapAgentV2 ---

    # 1. Script Generation Pipeline (Sequential)
    #    (Handles initial gen, refinement loop, post-processing)
    recap_refinement_loop = LoopAgent(
        name="RecapRefinementLoop",
        sub_agents=[recap_critic_agent, critique_processor_agent, recap_reviser_agent],
        max_iterations=1 # As in your original GameRecapAgent
    )
    recap_post_processing_sequence = SequentialAgent(
        name="RecapPostProcessing",
        sub_agents=[grammar_check_agent, tone_check_agent]
    )
    script_generation_pipeline_instance = SequentialAgent(
        name="ScriptGenerationPipeline",
        sub_agents=[
            query_setup_agent_instance, 
            initial_recap_generator_agent,
            game_state_setter_agent_instance, # <--- INSERTED HERE    # Output: current_recap
            entity_extractor_for_player_id_agent, # Extract teams for populator
            player_id_populator_agent_instance,   # Populate player_lookup_dict_json
            recap_refinement_loop,            # Modifies: current_recap
            recap_post_processing_sequence    # Modifies: grammar_suggestions, tone_check_result
        ],
        description="Generates, refines, and post-processes the game recap script. Primary output to state: 'current_recap'."
    )

    # 2. Audio Processing Pipeline (Sequential - TTS then STT)
    #    This will be one of the parallel branches.
    audio_processing_pipeline_for_parallel_instance = SequentialAgent(
        name="AudioProcessingPipelineForParallel", # Distinct name
        sub_agents=[
            text_to_speech_agent_instance,        # Reads 'current_recap', Output: 'generated_dialogue_audio_uri'
            speech_to_timestamps_agent_instance   # Reads 'generated_dialogue_audio_uri', Output: 'word_timestamps_list'
        ],
        description="Generates dialogue audio and then corresponding word timestamps."
    )


    # The GameRecapAgentV2 sequence needs to be adjusted:
    # 1. Script Generation
    # 2. Parallel Asset Initiation (the ParallelAgent above)
    # --- AT THIS POINT, THE AGENT CLIENT POLLING LOGIC IN main.py RUNS for all initiated LR tasks ---
    # 3. Result Processing and further dependent tasks:
    #    - Process Image LR results, then Critique & New Prompts (if desired in a loop)
    #    - Process Video LR results
    #    - Process TTS LR results, then Initiate STT LR task
    #    - (Client polls for STT LR task)
    #    - Process STT LR results
    # 4. AssetValidationAndRetryAgent (needs to be aware of the new state keys from LR tools)
    # 5. AssetAggregatorAgent

    # This makes GameRecapAgentV2 a more complex CustomAgent to manage these states and re-entrancy.
    # For this refactor, I will focus on getting the LR tools initiated and the client polling logic.
    # The subsequent processing will assume results are populated in state by the client handling.

    # Simplified sequence for GameRecapAgentV2 for this refactor:
    # Define placeholder "ResultProcessing" agents that would run after client polling.
    # These would read from state keys like "final_image_uris_from_lr_tool_json_string"
    # (which is the output_key of the LlmAgent that called the LRFT, after client injected FunctionResponse).

    image_final_result_processor = LlmAgent(
        name="ImageFinalResultProcessor", model=GEMINI_FLASH_MODEL_ID,
        instruction="""You are a data processor.
Read image URIs from session state '{image_generation_lr_tool_caller_agent_output_key}'.
Format them into 'all_generated_image_assets_details' and 'assets_for_critique_json'.
Output: 'Image results processed.'""",
        # Placeholder for actual state key name.
        output_key="image_final_result_processing_status"
    )
    video_final_result_processor = LlmAgent(
        name="VideoFinalResultProcessor", model=GEMINI_FLASH_MODEL_ID,
        instruction="""You are a data processor.
Read video URIs from session state '{video_generation_lr_tool_caller_agent_output_key}'.
Format them into 'final_video_assets_list'.
Output: 'Video results processed.'""",
        output_key="video_final_result_processing_status"
    )
  # == TTS REVISED (using LRFT) ==
    tts_lr_tool_caller_agent = LlmAgent(
        name="TTSLRToolCaller", model=GEMINI_FLASH_MODEL_ID,
        instruction="""Audio synthesis robot.
Read current recap from session state key 'current_recap'.
Read game PK from session state key 'game_pk'.
You MUST call the 'initiate_tts_generation' tool with the script and game_pk_str.
Your output will be the initial response from the tool (task details).""",
        tools=[long_running_tts_tool],
        output_key="tts_lr_task_submission_details" # Stores the initial {"status": "pending...", "task_id": ...}
    )

    # This agent is part of the main sequence AFTER client polling resolves the TTS task.
    tts_final_result_processor = LlmAgent(
        name="TTSFinalResultProcessor", model=GEMINI_FLASH_MODEL_ID,
        instruction="""You are a data processor.
The TTS generation tool has completed. Its output (a JSON string with 'audio_uri' or 'error')
is in session state key '{tts_lr_task_submission_details_resolved_output_key}'.
Parse this JSON string. If successful and an 'audio_uri' is present,
store this URI in session state key 'generated_dialogue_audio_uri'.
If there was an error or no URI, log it and ensure 'generated_dialogue_audio_uri' is None or reflects the error.
Output: 'TTS result processed, audio URI set.' or 'TTS processing failed.'""",
        # The input state key needs to be the output_key of tts_lr_tool_caller_agent
        # AFTER it has processed the injected FunctionResponse.
        # Let's rename tts_lr_tool_caller_agent.output_key to:
        # output_key="final_tts_details_from_lr_tool_json_string"
        # And update the instruction to use this.
        output_key="tts_final_result_processing_status"
    )
    # Modify tts_lr_tool_caller_agent.output_key:
    # tts_lr_tool_caller_agent.output_key = "final_tts_details_from_lr_tool_json_string"
    # And the instruction for tts_final_result_processor should use this new key.
    # This change would be:
    # instruction="...is in session state key 'final_tts_details_from_lr_tool_json_string'..."


    # == STT REVISED (using LRFT) ==
    stt_lr_tool_caller_agent = LlmAgent(
        name="STTLRToolCaller", model=GEMINI_FLASH_MODEL_ID,
        instruction="""Audio transcription robot.
Read the generated audio GCS URI from session state key 'generated_dialogue_audio_uri'.
If the URI is present, you MUST call the 'initiate_stt_transcription' tool with the audio_gcs_uri.
If no URI is present, output 'Skipping STT as no audio URI found.'
Your tool call output will be the initial response from the tool (task details).""",
        tools=[long_running_stt_tool],
        output_key="stt_lr_task_submission_details" # Stores initial {"status": "pending...", "task_id": ...}
    )

    stt_final_result_processor = LlmAgent(
        name="STTFinalResultProcessor", model=GEMINI_FLASH_MODEL_ID,
        instruction="""You are a data processor.
The STT transcription tool has completed. Its output (a JSON string list of word timestamps or an error)
is in session state key '{stt_lr_task_submission_details_resolved_output_key}'.
Parse this JSON string. If successful, store the list of timestamps in session state key 'word_timestamps_list'.
If there was an error, ensure 'word_timestamps_list' is empty or reflects the error.
Output: 'STT result processed, timestamps stored.' or 'STT processing failed.'""",
        # Similar to TTS, the input key here should be stt_lr_tool_caller_agent.output_key
        # after it processes the injected FunctionResponse. Let's call it:
        # "final_stt_timestamps_from_lr_tool_json_string"
        output_key="stt_final_result_processing_status"
    )

    # Update LlmAgent output keys for clarity when they are resolved by LRFT
    image_generation_lr_tool_caller_agent.output_key = "final_image_uris_from_lr_tool_json_string"
    video_generation_lr_tool_caller_agent.output_key = "final_video_uris_from_lr_tool_json_string"
    tts_lr_tool_caller_agent.output_key = "final_tts_details_from_lr_tool_json_string"
    stt_lr_tool_caller_agent.output_key = "final_stt_timestamps_from_lr_tool_json_string"

    # Update instructions for processor agents to use these new output_keys
    image_final_result_processor.instruction = image_final_result_processor.instruction.replace(
        '{image_generation_lr_tool_caller_agent_output_key}', image_generation_lr_tool_caller_agent.output_key
    )
    video_final_result_processor.instruction = video_final_result_processor.instruction.replace(
        '{video_generation_lr_tool_caller_agent_output_key}', video_generation_lr_tool_caller_agent.output_key
    )
    tts_final_result_processor.instruction = tts_final_result_processor.instruction.replace(
        '{tts_lr_tool_caller_agent_output_key}', tts_lr_tool_caller_agent.output_key
    )
    stt_final_result_processor.instruction = stt_final_result_processor.instruction.replace(
        '{stt_lr_tool_caller_agent_output_key}', stt_lr_tool_caller_agent.output_key
    )

    # --- Update ParallelAssetPipelinesAgent ---
    parallel_asset_pipelines_instance = ParallelAgent(
        name="ParallelAssetGenerationPipelines",
        sub_agents=[
            static_asset_pipeline_agent_instance,         # Unchanged in its LR nature (uses LlmAgents for tools)
            image_generation_initiation_sequence,         # NEW: Initiates LR image gen
            video_generation_initiation_sequence,         # NEW: Initiates LR video gen
            tts_lr_tool_caller_agent,      
            graph_generation_pipeline_agent_instance      # Unchanged
        ],
        description="Concurrently initiates generation of static assets, images, videos, audio (TTS), and graphs."
    )


    asset_aggregator_agent_instance = AssetAggregatorAgent(name="AssetAggregator") # Defined above
    # New GameRecapAgentV2 structure
    game_recap_assistant_v2 = SequentialAgent(
        name="game_recap_assistant_v2",
        sub_agents=[
            script_generation_pipeline_instance,       # Generates script, game_pk, prompts for assets
            parallel_asset_pipelines_instance,         # Initiates LR tasks for image, video, TTS, graph
            # --- Client Polling Happens Here for Image, Video, TTS, Graph LR tasks ---
            image_final_result_processor,              # Processes results of image LR task
            video_final_result_processor,              # Processes results of video LR task
            tts_final_result_processor,        # Processes TTS results, sets 'generated_dialogue_audio_uri'
            stt_lr_tool_caller_agent,          # Initiates STT LR task (uses 'generated_dialogue_audio_uri')
            # Client polling happens again for STT
            stt_final_result_processor,        # Processes STT results, sets 'word_timestamps_list'
            visual_critic_agent,                       # Critiques images based on results from image_final_result_processor
            new_visual_prompts_from_critique_agent,    # Creates new prompts (not used in a loop in this simplified version)
           # asset_validator_instance,                  # Validates all collected assets
            asset_aggregator_agent_instance            # Aggregates for final output
        ],
        description="Orchestrates game recap and ADK-native long-running asset creation with client-side polling."
    )




    # 5. The New GameRecapAgentV2 (Sequential Orchestrator)
    #    This is the agent that will be routed to by the root_agent.
    #    Its _run_async_impl is implicitly handled by SequentialAgent.
    #    We must provide a final output if this agent is called directly by the root agent (final response).
    #    To do this, we can add a final LlmAgent that just takes 'current_recap' and outputs it.





    # Inside game_recap_assistant_v2, a dummy _run_async_impl is needed if it inherits BaseAgent directly.
    # However, SequentialAgent, ParallelAgent, LoopAgent are workflow agents; they don't need _run_async_impl.
    # If game_recap_assistant_v2 itself needs to yield the final recap, it can't be just a SequentialAgent.
    # The SequentialAgent will yield events from its sub-agents. The final event from `final_recap_output_agent` will be the recap.

    # --- Root Agent ---
    root_agent = LlmAgent(
        model=MODEL_ID,
        name="ai_assistant",
        instruction=ROOT_AGENT_INSTRUCTION, # Make sure this routes to "game_recap_assistant_v2"
        sub_agents=[
            cocktail_agent,
            booking_agent,
            mlb_assistant,
            game_recap_assistant_v2 # Add the new parallelized game recap agent
        ],
    )
    return root_agent



# --- Agent Execution Helpers - MODIFIED for LRFT Polling ---
# Store pending long-running tasks that the client needs to poll
# Key: session_id, Value: List of task_ids_details e.g. [{"task_id": "...", "tool_name": "...", "original_tool_call_id": "..."}]
PENDING_LR_TASKS_FOR_CLIENT_POLLING: Dict[str, List[Dict[str,Any]]] = {}

async def _handle_long_running_tool_event(event: Event, session_id: str, user_id: str, runner: Runner):
    if event.long_running_tool_ids and event.content and event.content.parts:
        for part in event.content.parts:
            if part.function_call and part.function_call.id in event.long_running_tool_ids:
                # This is the initial FunctionCall to our LongRunningFunctionTool
                fc_response_content = part.function_call.response.content if part.function_call.response else None # type: ignore
                if fc_response_content:
                    try:
                        # The initial response from our LRFT initiator (e.g., initiate_image_generation)
                        # should contain {"status": "pending_agent_client_action", "task_id": ..., "tool_name": ...}
                        initial_tool_output = json.loads(fc_response_content)
                        task_id = initial_tool_output.get("task_id")
                        tool_name = initial_tool_output.get("tool_name")
                        original_tool_call_id = part.function_call.id # ID of the call to the LRFT

                        if initial_tool_output.get("status") == "pending_agent_client_action" and task_id and tool_name:
                            logger.info(f"LR_CLIENT: Detected pending LR task {task_id} for tool {tool_name} (call_id: {original_tool_call_id}). Will poll.")
                            if session_id not in PENDING_LR_TASKS_FOR_CLIENT_POLLING:
                                PENDING_LR_TASKS_FOR_CLIENT_POLLING[session_id] = []
                            PENDING_LR_TASKS_FOR_CLIENT_POLLING[session_id].append({
                                "task_id": task_id,
                                "tool_name": tool_name,
                                "original_tool_call_id": original_tool_call_id,
                                "user_id": user_id # Store user_id for subsequent run_async
                            })
                            # Do not yield this event's text to user yet, as the task is not complete.
                            return True # Indicates an LR task was detected and is being handled
                    except json.JSONDecodeError:
                        logger.error(f"LR_CLIENT: Could not parse initial response from LRFT: {fc_response_content}")
                    except Exception as e:
                        logger.error(f"LR_CLIENT: Error processing LRFT initial response: {e}")
    return False


from google.genai import types as genai_types # Explicit import for clarity

def get_lr_function_call_if_any(event: Event, lr_tool_names: List[str]) -> Optional[genai_types.FunctionCall]:
    """
    Checks if the event contains a FunctionCall to one of the specified LongRunningFunctionTool names.
    event.long_running_tool_ids signals that a call *to an LRFT* was made.
    We also check the name to ensure it's one we are managing with client-side polling.
    """
    if not event.long_running_tool_ids or not event.content or not event.content.parts:
        return None
    for part in event.content.parts:
        if (
            part.function_call
            and part.function_call.id in event.long_running_tool_ids
            and part.function_call.name in lr_tool_names
        ):
            logger.debug(f"LR_HELPER: Identified LR FunctionCall: ID={part.function_call.id}, Name={part.function_call.name}")
            return part.function_call
    return None

def get_lr_initiator_function_response(event: Event, original_call_id: str) -> Optional[genai_types.FunctionResponse]:
    """
    Gets the FunctionResponse from the LRFT's initiator function, matching the original_call_id.
    """
    if not event.content or not event.content.parts:
        return None
    for part in event.content.parts:
        if (
            part.function_response
            and part.function_response.id == original_call_id
        ):
            logger.debug(f"LR_HELPER: Identified LR Initiator FunctionResponse: ID={part.function_response.id}, Name={part.function_response.name}")
            return part.function_response
    return None


# --- Agent Execution Helpers ---
async def _run_agent_and_get_response(
    runner: Runner,
    session_id: str,
    content: types.Content,
    # Add user_id for consistency with LR polling needs
    user_id: str,
        # List of your LRFT tool names that require client polling
    managed_lr_tool_names: List[str] = [
        "initiate_image_generation",
        "initiate_video_generation",
        "initiate_tts_generation", # Add TTS initiator name
        "initiate_stt_transcription"  # Add STT initiator name
    ]
) -> List[str]:
    logger.info(f"AGENT_RUN: Starting for session {session_id}, user {user_id}. Initial message: {content.parts[0].text if content.parts else 'no text'}")
    response_parts: List[str] = []
    
    # Variables to track the state of handling one LRFT at a time per "turn"
    # A single user message might lead to an agent turn that calls one LRFT.
    # We process that before looking for another.
    current_lr_function_call: Optional[genai_types.FunctionCall] = None
    # current_lr_function_response stores the *initiator's* response (task_id, etc.)
    current_lr_function_response: Optional[genai_types.FunctionResponse] = None

    # Initial agent run with the new message
    current_events_stream = runner.run_async(
        session_id=session_id, user_id=user_id, new_message=content
    )

    processing_complete_for_this_turn = False
    while not processing_complete_for_this_turn:
        async for event in current_events_stream:
            if event.content and event.content.parts:
                for part in event.content.parts:
                    # 1. Collect model's textual responses
                    if part.text:
                        # Suppress pending messages from initiator if desired, or log them
                        if "pending_agent_client_action" not in part.text and \
                           "Awaiting client polling" not in part.text:
                            response_parts.append(part.text)
                            logger.info(f"AGENT_RUN [{event.author or 'agent'}]: Appended text: \"{part.text[:100]}...\"")
                        else:
                            logger.info(f"AGENT_RUN [{event.author or 'agent'}]: LRFT pending message: \"{part.text}\"")
                    
                    # 2. Detect FunctionCall to our LRFTs
                    if not current_lr_function_call: # Only look for a new LR call if not already tracking one
                        fc = get_lr_function_call_if_any(event, managed_lr_tool_names)
                        if fc:
                            current_lr_function_call = fc
                            logger.info(f"AGENT_RUN: Detected call to LRFT '{fc.name}' (ID: {fc.id}). Waiting for initiator's response.")
                            # Don't break; initiator response might be in the same event or next part.

                    # 3. Get the initiator's FunctionResponse for the detected LRFT call
                    if current_lr_function_call and not current_lr_function_response:
                        fr = get_lr_initiator_function_response(event, current_lr_function_call.id)
                        if fr:
                            current_lr_function_response = fr
                            logger.info(f"AGENT_RUN: Received initiator response for LRFT '{current_lr_function_call.name}' (ID: {current_lr_function_call.id}).")
                            try:
                                # The initiator function (e.g., initiate_image_generation) returns a dict.
                                # This dict is placed in fr.response by the ADK.
                                initiator_output = fr.response # This should be a dict
                                if not isinstance(initiator_output, dict):
                                     # Sometimes it might be a string that needs parsing, depending on ADK version or LLM interaction
                                     if isinstance(initiator_output, str):
                                         initiator_output = json.loads(initiator_output)
                                     else:
                                         raise ValueError(f"Initiator output is not a dict or parsable string: {type(initiator_output)}")


                                task_id = initiator_output.get("task_id")
                                tool_name = initiator_output.get("tool_name") # Get tool_name from initiator's payload

                                if initiator_output.get("status") == "pending_agent_client_action" and task_id and tool_name:
                                    logger.info(f"AGENT_RUN: Adding LR task to PENDING_LR_TASKS_FOR_CLIENT_POLLING: task_id={task_id}, tool_name={tool_name}, original_call_id={current_lr_function_call.id}")
                                    if session_id not in PENDING_LR_TASKS_FOR_CLIENT_POLLING:
                                        PENDING_LR_TASKS_FOR_CLIENT_POLLING[session_id] = []
                                    PENDING_LR_TASKS_FOR_CLIENT_POLLING[session_id].append({
                                        "task_id": task_id,
                                        "tool_name": tool_name, # Use tool_name from payload
                                        "original_tool_call_id": current_lr_function_call.id,
                                        "user_id": user_id
                                    })
                                else:
                                    logger.warning(f"AGENT_RUN: LRFT initiator for '{current_lr_function_call.name}' did not return 'pending_agent_client_action' status or missing task_id/tool_name. Payload: {initiator_output}")
                            except Exception as e:
                                logger.error(f"AGENT_RUN: Error processing initiator output for LRFT '{current_lr_function_call.name}': {e}. Response payload: {fr.response}", exc_info=True)
                            
                            # Reset for the next potential LRFT call within the same agent turn (if any)
                            # Though typically one LRFT call means the agent pauses.
                            current_lr_function_call = None
                            current_lr_function_response = None
        
        # After processing all events from the current stream:
        if session_id in PENDING_LR_TASKS_FOR_CLIENT_POLLING and PENDING_LR_TASKS_FOR_CLIENT_POLLING[session_id]:
            logger.info(f"AGENT_RUN: Polling for session {session_id} as tasks are pending.")
            # _poll_and_inject_responses will call runner.run_async internally for injections.
            # We need a new stream for *after* those injections.
            await _poll_and_inject_responses(session_id, runner)

            if session_id in PENDING_LR_TASKS_FOR_CLIENT_POLLING and PENDING_LR_TASKS_FOR_CLIENT_POLLING[session_id]:
                logger.info(f"AGENT_RUN: Tasks still pending for {session_id} after poll cycle. Agent may need more turns or is waiting for these.")
                # If an agent is designed to make multiple LRFT calls sequentially, or if one LRFT
                # resolution triggers another, the loop needs to continue.
                # We create a new stream by calling run_async with no new user message,
                # allowing the agent to process the injected FunctionResponses.
                current_events_stream = runner.run_async(
                    session_id=session_id, user_id=user_id, new_message=None # No new user message
                )
                # response_parts = [] # Optionally clear response parts if only accumulating final turn's response
            else:
                logger.info(f"AGENT_RUN: All pending tasks for {session_id} resolved in this poll cycle.")
                # Get the agent's response after all polling and injections for this turn.
                current_events_stream = runner.run_async(
                    session_id=session_id, user_id=user_id, new_message=None
                )
                # Let the loop run one more time with this new stream to collect final text.
                # The loop will then break if no new LR tasks are initiated.
        else:
            logger.info(f"AGENT_RUN: No pending LR tasks for session {session_id} to poll. Turn complete.")
            processing_complete_for_this_turn = True # Exit the while loop

    logger.info(f"AGENT_RUN: Finished for session {session_id}. Final response parts count: {len(response_parts)}")
    return response_parts



async def _poll_and_inject_responses(session_id: str, runner: Runner):
    if session_id not in PENDING_LR_TASKS_FOR_CLIENT_POLLING or not PENDING_LR_TASKS_FOR_CLIENT_POLLING[session_id]:
        return

    logger.debug(f"POLL_INJECT: Starting poll cycle for session {session_id}. Pending tasks: {len(PENDING_LR_TASKS_FOR_CLIENT_POLLING[session_id])}")
    tasks_to_re_add = [] # For tasks that are submitted but not yet completed/failed
    processed_indices_in_this_poll = []


    for idx, task_details in enumerate(PENDING_LR_TASKS_FOR_CLIENT_POLLING[session_id]):
        task_id = task_details["task_id"]
        tool_name = task_details["tool_name"]
        original_tool_call_id = task_details["original_tool_call_id"]
        user_id_for_task = task_details["user_id"]

        task_status_obj = None
        final_result_payload_str = None # This should be the JSON string

        # Determine which global task list to check
        if tool_name == "initiate_image_generation": # Match the tool_name from initiator payload
            task_status_obj = IMAGE_GENERATION_TASKS.get(task_id)
        elif tool_name == "initiate_video_generation":
            task_status_obj = VIDEO_GENERATION_TASKS.get(task_id)
        # Add elif for TTS, STT tasks here, checking their respective global task dicts
        elif tool_name == "initiate_tts_generation":
            task_status_obj = TTS_GENERATION_TASKS.get(task_id)
        elif tool_name == "initiate_stt_transdcription":  
             task_status_obj = STT_TRANSCRIPTION_TASKS.get(task_id)


        if task_status_obj:
            current_status = task_status_obj.get("status")
            if current_status in ["completed", "failed"]:
                logger.info(f"POLL_INJECT: Task {task_id} ({tool_name}) fully completed with status: {current_status}.")
                if current_status == "completed":
                    # 'result' from our background task is already a JSON string.
                    final_result_payload_str = task_status_obj.get("result", '"[]"') # Default to empty JSON list string
                else: # failed
                    final_result_payload_str = json.dumps({"error": task_status_obj.get("error", "Unknown error in task")})
                
                # The `response` for FunctionResponse must be a dict.
                # If the LlmAgent that called the LRFT is simple and its instruction is to output the tool's result directly,
                # and the "result" is the JSON string, we package it as {"result": "json_string_payload"}.
                # This way, the LlmAgent's output_key will store the raw JSON string.
                function_response_content_dict = {"result": final_result_payload_str}

                function_response_part = genai_types.Part(
                    function_response=genai_types.FunctionResponse(
                        id=original_tool_call_id,
                        name=tool_name, # Name of the original LRFT initiator tool
                        response=function_response_content_dict
                    )
                )
                logger.info(f"POLL_INJECT: Injecting FunctionResponse for task {task_id} (call_id: {original_tool_call_id}) into session {session_id}.")
                
                # Create a new stream for this injection.
                # The consumer of _poll_and_inject_responses will then get the next stream.
                injection_event_stream = runner.run_async(
                    session_id=session_id,
                    user_id=user_id_for_task,
                    new_message=genai_types.Content(parts=[function_response_part], role='tool') # role='tool' is correct for FunctionResponse
                )
                async for event in injection_event_stream: # Consume events from this injection run
                    logger.debug(f"POLL_INJECT: Event from injection run for {task_id}: {event.type}")
                    # Potentially, this injection run could trigger another LRFT.
                    # Check for new LRFT calls resulting from this injection.
                    # This is where it gets recursive if not careful.
                    # For now, assume injection primarily provides data back.
                    pass


                processed_indices_in_this_poll.append(idx)
                # Clean up from global tracking dicts
                if tool_name == "initiate_image_generation" and task_id in IMAGE_GENERATION_TASKS: del IMAGE_GENERATION_TASKS[task_id]
                elif tool_name == "initiate_video_generation" and task_id in VIDEO_GENERATION_TASKS: del VIDEO_GENERATION_TASKS[task_id]
                elif tool_name == "initiate_tts_generation" and task_id in TTS_GENERATION_TASKS: del TTS_GENERATION_TASKS[task_id] # NEW
                elif tool_name == "initiate_stt_transcription" and task_id in STT_TRANSCRIPTION_TASKS: del STT_TRANSCRIPTION_TASKS[task_id] # NEW
                # Add TTS/STT cleanup here

            elif current_status == "submitted" or current_status == "processing":
                logger.debug(f"POLL_INJECT: Task {task_id} ({tool_name}) still '{current_status}'. Will poll again later.")
                # tasks_to_re_add.append(task_details) # No, just leave it in the main list
            else: # Unknown status
                logger.warning(f"POLL_INJECT: Task {task_id} ({tool_name}) has unknown status: {current_status}. Removing from polling.")
                processed_indices_in_this_poll.append(idx)

        else: # Task not found in its tracking dictionary (should not happen if added correctly)
            logger.warning(f"POLL_INJECT: Task {task_id} ({tool_name}) not found in its tracking dictionary. Removing from polling.")
            processed_indices_in_this_poll.append(idx)
    
    # Remove processed tasks from PENDING_LR_TASKS_FOR_CLIENT_POLLING
    if processed_indices_in_this_poll:
        new_pending_list = [
            item for i, item in enumerate(PENDING_LR_TASKS_FOR_CLIENT_POLLING[session_id])
            if i not in processed_indices_in_this_poll
        ]
        if not new_pending_list:
            del PENDING_LR_TASKS_FOR_CLIENT_POLLING[session_id]
            logger.debug(f"POLL_INJECT: All tasks for session {session_id} removed from pending list.")
        else:
            PENDING_LR_TASKS_FOR_CLIENT_POLLING[session_id] = new_pending_list
            logger.debug(f"POLL_INJECT: Session {session_id} now has {len(new_pending_list)} tasks remaining in pending list.")


# NEW Helper specifically for the voice query path, ensuring user_id is passed
async def _run_voice_agent_and_get_response(
    runner: Runner,
    session_id: str,
    user_id: str, # Explicitly take user_id
    content: types.Content,
) -> List[str]:
    logging.info(f"VOICE AGENT HELPER: Running agent for app '{runner.app_name}', session '{session_id}', user '{user_id}'.")
    events_async = runner.run_async(
        session_id=session_id,
        user_id=user_id,     # Pass user_id explicitly
        new_message=content
        # app_name is known by the runner instance
    )
    response_parts: List[str] = []
    async for event in events_async:
        try:
            if hasattr(event, "content") and event.content.role == "model":
                if hasattr(event.content, "parts") and event.content.parts:
                    if event.content.parts[0] and hasattr(event.content.parts[0], "text"):
                        part_text = getattr(event.content.parts[0], "text", None)
                        if isinstance(part_text, str) and part_text:
                            response_parts.append(part_text)
        except AttributeError as e:
            logging.warning("Could not process event attribute during voice agent helper run: %s. Event: %s", e, event)
    logging.info(f"VOICE AGENT HELPER: Agent run finished. Response parts: {response_parts}")
    return response_parts


# main.py
# ... (imports)
_run_agent_and_get_response
# --- Global variable for monkey-patching ---
original_get_session = None # Defined globally

# ... (APP_NAME, session_service, loaded_mcp_tools_global, etc.) ...

async def process_voice_query_with_adk(session_id: str, user_query: str) -> str:
    global original_get_session # <<< --- ADD THIS LINE ---
    
    logging.info(f"ADK processing voice query for session {session_id}: '{user_query}'")
    user_id_for_adk = session_id
    session_key_tuple = (APP_NAME, user_id_for_adk, session_id)

    if not loaded_mcp_tools_global: return "Core tools not available."
    if not session_service: return "Session service not available."

    try:
        # 1. Ensure ADK session exists (create if not)
        current_session_obj = None
        retrieved_session_from_get = None

        logging.info(f"VOICE_ADK: Attempting to GET session with key components: app='{APP_NAME}', user='{user_id_for_adk}', session='{session_id}'")
        try:
            retrieved_session_from_get = session_service.get_session(
                app_name=APP_NAME, user_id=user_id_for_adk, session_id=session_id
            )
        except KeyError:
            logging.info(f"VOICE_ADK: get_session raised KeyError for {session_key_tuple}. Session does not exist yet.")
            # retrieved_session_from_get remains None

        if retrieved_session_from_get is not None:
            current_session_obj = retrieved_session_from_get
            logging.info(f"VOICE_ADK: Session GET successful. Session ID: {current_session_obj.id}, State: {current_session_obj.state}")
        else:
            logging.info(f"VOICE_ADK: Session not found/None via GET. Attempting to CREATE session with key: {session_key_tuple}")
            current_session_obj = session_service.create_session(
                app_name=APP_NAME, user_id=user_id_for_adk, session_id=session_id, state={"source": "voice_adk_create_after_get_fail"}
            )
            logging.info(f"VOICE_ADK: Session CREATED successfully. Session ID: {current_session_obj.id}, State: {current_session_obj.state}")
        
        if hasattr(session_service, '_sessions') and isinstance(session_service._sessions, dict):
            logging.info(f"VOICE_ADK: Keys in session_service._sessions before Runner: {list(session_service._sessions.keys())}")
            if session_key_tuple in session_service._sessions:
                actual_obj_in_dict = session_service._sessions[session_key_tuple]
                obj_id = actual_obj_in_dict.id if actual_obj_in_dict else 'None Object in Dict'
                logging.info(f"VOICE_ADK: CONFIRMED key {session_key_tuple} IS in _sessions. Object type: {type(actual_obj_in_dict)}, ID: {obj_id}")
            else:
                logging.warning(f"VOICE_ADK: WARNING - key {session_key_tuple} IS NOT in _sessions dict before Runner.")
        
        content_for_adk = types.Content(role="user", parts=[types.Part(text=user_query)])
        root_adk_agent = await create_agent_with_preloaded_tools(loaded_mcp_tools_global)

        logging.info(f"VOICE_ADK: Initializing Runner with app_name='{APP_NAME}', agent_type={type(root_adk_agent)}, artifact_service_type={type(artifacts_service)}, session_service_type={type(session_service)}")
        
        # --- RUNNER INITIALIZATION ---
        adk_runner = Runner(
            app_name=APP_NAME,
            agent=root_adk_agent,
            artifact_service=artifacts_service,
            session_service=session_service
            # memory_service is optional in your Runner's signature, so omitting it is fine
        )
        logging.info(f"VOICE_ADK: Runner initialized successfully. Instance ID: {id(adk_runner)}")
        # --- END OF CORRECTION ---
        
        logging.info(f"VOICE_ADK: Calling _run_voice_agent_and_get_response with session_id='{session_id}', user_id='{user_id_for_adk}'")
        response_parts = await _run_voice_agent_and_get_response(
            adk_runner, session_id, user_id_for_adk, content_for_adk
        )
        # The restoration of original_get_session will happen in the finally block

        if response_parts:
            final_response = " ".join(response_parts)
            return final_response
        else:
            return "ADK processed (voice) but no text parts."

    # ... (except ValueError, except Exception as before) ...
    except ValueError as ve:
        # ...
        logging.error(f"VOICE_ADK: ValueError: {ve}", exc_info=True)
        return "I'm having trouble with conversation data (Runner)." if "Session not found" in str(ve) else "Data problem (voice)."
    except Exception as e:
        logging.error(f"VOICE_ADK: General error: {e}", exc_info=True)
        return "Unexpected problem (voice)."
    finally:
        # Ensure restoration of the original get_session method
        # Check if it was patched in THIS call by checking if original_get_session (the global) has a value
        if original_get_session is not None and hasattr(session_service, 'get_session') and session_service.get_session.__name__ == 'patched_get_session':
            session_service.get_session = original_get_session
            logging.info("VOICE_ADK: Restored original session_service.get_session in finally block.")
            original_get_session = None # Reset global for the next potential call to process_voice_query_with_adk

async def _get_runner_async(
    loaded_mcp_tools: Dict[str, Any], session_id: str, query: str, user_id: str 
) -> List[str]:
    """
    Sets up and runs the root agent for a given query using preloaded tools.

    This function creates a root agent, initializes a runner, and then
    executes the agent with the user's query, returning the response parts.

    Args:
        loaded_mcp_tools: A dictionary of pre-loaded MCP tools.
        session_id: The unique identifier for the user's session.
        query: The user's input query as a string.

    Returns:
        A list of strings representing the parts of the agent's textual response.
        Returns an error message list if critical services are unavailable.
    """
    content = types.Content(role="user", parts=[types.Part(text=query)])

    if artifacts_service is None or session_service is None:
        logging.error(
            "Artifact or Session service is not initialized for _get_runner_async."
        )
        return ["Error: Core services not available."]

    if not loaded_mcp_tools:
        logging.error("MCP tools are not available for _get_runner_async.")
        return ["Error: Essential tools not loaded, cannot process request."]

    root_agent = await create_agent_with_preloaded_tools(loaded_mcp_tools)
    logging.info(f"TEXT_PATH: About to create Runner. Using session_service instance ID: {id(session_service)}")
    logging.info(f"TEXT_PATH: Session service type: {type(session_service)}")
    if hasattr(session_service, '_sessions'):
        logging.info(f"TEXT_PATH: Keys in session_service._sessions before Runner: {list(session_service._sessions.keys())}")
    else:
        logging.info("TEXT_PATH: session_service does not have _sessions attribute.")
    runner = Runner(
        app_name=APP_NAME,
        agent=root_agent,
        artifact_service=artifacts_service,
        session_service=session_service,
    )
    logging.info(f"TEXT_PATH: Runner created. Runner's session_service instance ID: {id(runner.session_service)}")
    response = await _run_agent_and_get_response(runner, session_id, content, user_id)
    return response


# --- FastAPI Application ---


@asynccontextmanager
async def app_lifespan(app_instance: FastAPI) -> Any: 
    """
    Manages application startup and shutdown operations for the FastAPI app.
    Args:
        app_instance: The FastAPI application instance.
    """
    global loaded_mcp_tools_global # To store tools for the voice agent call
    logging.info("Application Lifespan: Startup initiated.")
    app_instance.state.mcp_tools = {}
    app_instance.state.mcp_tool_exit_stack = None

    try:
        collected_tools, tool_stack = await _collect_tools_stack(
            server_configs_instance
        )
        app_instance.state.mcp_tools = collected_tools
        loaded_mcp_tools_global = collected_tools     # For voice agent function
        app_instance.state.mcp_tool_exit_stack = tool_stack
        logging.info(
            "Application Lifespan: MCP Toolset initialized. Tools: %s",
            list(app_instance.state.mcp_tools.keys()),
        )
    except FileNotFoundError as file_error:
        logging.error("Command or script not found for key: %s", file_error)
    except ConnectionRefusedError as conn_refused:
        logging.error("Connection refused for key: %s", conn_refused)
    except Exception as e: # Catch a broader range of exceptions during startup
        logging.error(f"Critical error during MCP Toolset initialization: {e}", exc_info=True)
    yield

    logging.info("Application Lifespan: Shutdown initiated.")
    if app_instance.state.mcp_tool_exit_stack:
        logging.info("Application Lifespan: Closing MCP Toolset connections.")
        try:
            await app_instance.state.mcp_tool_exit_stack.aclose()
            logging.info(
                "Application Lifespan: MCP Toolset connections closed successfully."
            )
        except ConnectionRefusedError as conn_refused:
            logging.error("Connection refused for key: %s", conn_refused)
    else:
        logging.warning(
            "Application Lifespan: No MCP Toolset exit stack found to close."
        )


# Instantiate FastAPI with the lifespan manager
app = FastAPI(lifespan=app_lifespan)

# Include the Voice Agent Router
app.include_router(voice_agent_router) # The paths from voice_agent.py will be registered

# --- WebSocket Communication ---
async def run_adk_agent_async(
    websocket: WebSocket, loaded_mcp_tools: Dict[str, Any], session_id: str, user_id: str
) -> None:
    """
    Handles the continuous WebSocket communication loop for a connected client.

    Receives text messages from the client, processes them using the ADK agent
    (via `_get_runner_async`), and sends the agent's responses back to the client.

    Args:
        websocket: The WebSocket connection object for the client.
        loaded_mcp_tools: Pre-loaded MCP tools for the agent.
        session_id: The unique identifier for the client's session.
    """
    try:
        while True:
            text = await websocket.receive_text()
            # Pass user_id to _get_runner_async
            response_parts = await _get_runner_async(loaded_mcp_tools, session_id, text, user_id)

            if not response_parts and not (session_id in PENDING_LR_TASKS_FOR_CLIENT_POLLING and PENDING_LR_TASKS_FOR_CLIENT_POLLING[session_id]):
                # No text and no pending LR tasks for this session means agent is truly idle or finished.
                # If there are pending tasks, the client is handling them, no text response yet.
                logger.info("Agent for session %s did not produce text and no LR tasks pending after full processing.", session_id)
                # Optionally send a "Thinking..." or no message.
                # For now, only send if there are actual response parts.
                # continue # Commented out to allow sending empty if no parts and no pending.

            # Only send message if there are parts. Suppress if only LR polling is happening.
            if response_parts:
                ai_message = "\n".join(response_parts)
                await websocket.send_text(json.dumps({"message": ai_message}))
            elif not (session_id in PENDING_LR_TASKS_FOR_CLIENT_POLLING and PENDING_LR_TASKS_FOR_CLIENT_POLLING[session_id]):
                # If no response parts AND no pending LR tasks, send a generic ack or empty message if required by protocol
                await websocket.send_text(json.dumps({"message": ""})) # Example: send empty if no text

            await asyncio.sleep(0) # Yield control

    except WebSocketDisconnect:
        logger.info("Client %s disconnected from run_adk_agent_async.", session_id)
    finally:
        # Clean up any pending tasks for this session if the WebSocket disconnects
        if session_id in PENDING_LR_TASKS_FOR_CLIENT_POLLING:
            logger.info(f"Cleaning up PENDING_LR_TASKS_FOR_CLIENT_POLLING for disconnected session {session_id}")
            del PENDING_LR_TASKS_FOR_CLIENT_POLLING[session_id]
        # Also cancel any actual asyncio background tasks associated with this session if possible (harder to track without more robust task management)
        logger.info("Agent WebSocket task ending for session %s.", session_id)
 
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str) -> None:
    """
    FastAPI WebSocket endpoint for client connections.

    Accepts new WebSocket connections, creates a user session, and starts
    the `run_adk_agent_async` task to handle communication with the ADK agent.
    Ensures that MCP tools are loaded before starting the agent task.

    Args:
        websocket: The WebSocket connection object.
        session_id: The unique session identifier passed in the URL path.
    """
    await websocket.accept()
    user_id = session_id # Using session_id as user_id for this example
    logging.info("Client %s connected to WebSocket endpoint.", session_id)
    try:
        session_service.create_session(
            app_name=APP_NAME, user_id=session_id, session_id=session_id, state={}
        )

        # Access tools from app.state (set by the lifespan manager)
        loaded_mcp_tools = websocket.app.state.mcp_tools
        mcp_stack_exists = websocket.app.state.mcp_tool_exit_stack is not None

        if not loaded_mcp_tools or not mcp_stack_exists:
            logging.error(
                "MCP Tools not properly initialized. Cannot serve requests for session %s.",
                session_id,
            )
            await websocket.send_text(
                json.dumps(
                    {
                        "message": "Error: Server is not fully initialized. Please try again later."
                    }
                )
            )
            await websocket.close(code=1011)
            return

        await run_adk_agent_async(websocket, loaded_mcp_tools, session_id, user_id)

    except WebSocketDisconnect:
        logging.info(
            "Client %s disconnected from websocket_endpoint (early).", session_id
        )
    finally:
        logging.info("WebSocket endpoint cleanup for session %s.", session_id)


# --- TEMPORARY TEST ROUTE (Remove or comment out for production) ---
class TestADKQuery(BaseModel):
    session_id: str
    query: str

@app.post("/test_adk_voice_logic")
async def test_adk_processing(payload: TestADKQuery):
    """
    Temporary HTTP endpoint to directly test process_voice_query_with_adk.
    """
    logging.info(f"Received test request for /test_adk_voice_logic with payload: {payload}")
    if not loaded_mcp_tools_global: # Check if lifespan has run and tools are loaded
         raise HTTPException(status_code=503, detail="ADK Tools not loaded yet. Wait for app startup.")
    try:
        response_text = await process_voice_query_with_adk(
            session_id=payload.session_id,
            user_query=payload.query
        )
        return {"session_id": payload.session_id, "query": payload.query, "adk_response": response_text}
    except Exception as e:
        logging.error(f"Error in /test_adk_voice_logic: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
# --- END OF TEMPORARY TEST ROUTE ---


# Mount static files (e.g., for a web UI)
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")