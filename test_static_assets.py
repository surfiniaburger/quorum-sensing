import asyncio
import contextlib
from contextlib import asynccontextmanager
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, AsyncGenerator
import hashlib

from fastapi import FastAPI, HTTPException, Path
from pydantic import BaseModel, Field
from typing_extensions import override

# ADK Imports
from google.adk.agents import LlmAgent, BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService, Session # Import Session
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters
from google.genai import types
from google.adk.events import Event

# --- Configuration & Global Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

ISOLATED_APP_NAME = "StaticAssetTesterApp"
MODEL_ID = "gemini-2.0-flash"
GEMINI_FLASH_MODEL_ID="gemini-2.5-flash-preview-04-17"

isolated_loaded_mcp_tools: Dict[str, Any] = {}
isolated_session_service = InMemorySessionService()
isolated_artifacts_service = InMemoryArtifactService()


# --- Helper Function ---
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

class AllServerConfigs(BaseModel):
    configs: Dict[str, StdioServerParameters]

static_retriever_server_params_isolated = StdioServerParameters(
    command="python",
    args=["./mcp_server/static_asset_retriever_mcp_server.py"],
)
image_embedding_server_params_isolated = StdioServerParameters(
    command="python",
    args=["./mcp_server/image_embedding_server.py"],
)

server_configs_instance_isolated = AllServerConfigs(
    configs={
        "static_retriever_mcp": static_retriever_server_params_isolated,
        "image_embedding_mcp": image_embedding_server_params_isolated,
    }
)

# --- LlmAgent definitions (those NOT needing MCP tools immediately) ---
entity_extractor_for_assets_agent = LlmAgent(
    name="EntityExtractorForAssets",
    model=GEMINI_FLASH_MODEL_ID,
    instruction="""
You are a text analysis assistant.
Read the dialogue script provided in session state key 'current_recap'.
Identify all unique full player names (e.g., "Willy Adames", "Shohei Ohtani") and unique full MLB team names (e.g., "Los Angeles Angels", "Milwaukee Brewers") mentioned in the script.
The team names should be the full official names if present (e.g. "Los Angeles Dodgers", not just "Dodgers" if the full name appears). Prioritize full names.

Output a JSON object with two keys:
- "players": A list of unique player full name strings.
- "teams": A list of unique MLB team full name strings.

Example Input Script (from 'current_recap'):
"Host A: The Los Angeles Angels fought hard, but the New York Yankees were too strong today.
 Host B: Definitely, and Shohei Ohtani had a great game for the Angels, even in a loss. Aaron Judge, on the other hand, was unstoppable for the Yankees."

Example Output (as a JSON string):
{{
  "players": ["Shohei Ohtani", "Aaron Judge"],
  "teams": ["Los Angeles Angels", "New York Yankees"]
}}

If no players or teams are found, output empty lists within the JSON object (e.g., {"players": [], "teams": []}).
Ensure your entire output is a single, valid JSON string.
    """,
    output_key="extracted_entities_json"
)

static_asset_query_generator_for_assets_agent = LlmAgent(
    name="StaticAssetQueryGenerator",
    model=GEMINI_FLASH_MODEL_ID,
    instruction="""
You are an asset planner.
Expected in session state: 'extracted_entities_json', which is a JSON string like '{"players": ["Player A Full Name"], "teams": ["Full Team Name X"]}'.

Your task:
1. Parse the JSON string from 'extracted_entities_json' to get lists of player names and team names.
2. For each full team name in the "teams" list, generate a query string: "[Full Team Name] logo".
3. For each player full name in the "players" list, generate a query string: "[Player Full Name] headshot".
4. Combine all these generated query strings into a single list.
5. Output this final list of query strings as a single, valid JSON string.

Example Input ('extracted_entities_json'):
'{"players": ["Shohei Ohtani", "Aaron Judge"], "teams": ["Los Angeles Angels", "New York Yankees"]}'

Example Output (as a JSON string):
"[\\"Los Angeles Angels logo\\", \\"New York Yankees logo\\", \\"Shohei Ohtani headshot\\", \\"Aaron Judge headshot\\"]"

If the input 'extracted_entities_json' is empty or represents no entities (e.g., '{"players": [], "teams": []}'),
then output an empty JSON list string: "[]".
    """,
    output_key="static_asset_search_queries_json",
)

# --- StaticAssetPipelineAgent Definition ---
class StaticAssetPipelineAgent(BaseAgent):
    model_config = {"arbitrary_types_allowed": True}
    entity_extractor: LlmAgent
    static_asset_query_generator: LlmAgent
    logo_searcher_llm: LlmAgent
    headshot_retriever_llm: LlmAgent

    def __init__(self, name: str,
                 entity_extractor: LlmAgent,
                 static_asset_query_generator: LlmAgent,
                 logo_searcher_llm: LlmAgent,
                 headshot_retriever_llm: LlmAgent):
        super().__init__(
            name=name,
            entity_extractor=entity_extractor,
            static_asset_query_generator=static_asset_query_generator,
            logo_searcher_llm=logo_searcher_llm,
            headshot_retriever_llm=headshot_retriever_llm,
            sub_agents=[
                entity_extractor,
                static_asset_query_generator,
                logo_searcher_llm,
                headshot_retriever_llm
            ]
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
        logger.info(f"[{self.name}] Starting REFACTORED static asset retrieval pipeline.")
        ctx.session.state["current_static_assets_list"] = []

        current_recap = ctx.session.state.get("current_recap")
        if not current_recap:
            logger.warning(f"[{self.name}] 'current_recap' missing. Skipping static asset pipeline.")
            yield Event(author=self.name, content=types.Content(role="model", parts=[types.Part(text="Static asset pipeline skipped: no recap.")]))
            return

        logger.info(f"[{self.name}] Running EntityExtractorForAssets...")
        async for event in self.entity_extractor.run_async(ctx): yield event

        logger.info(f"[{self.name}] Running StaticAssetQueryGenerator...")
        async for event in self.static_asset_query_generator.run_async(ctx): yield event

        queries_json_str = ctx.session.state.get("static_asset_search_queries_json", "[]")
        logger.info(f"[{self.name}] Raw output from StaticAssetQueryGenerator: '{queries_json_str[:500]}...'")
        cleaned_queries_json_str = _clean_json_string_from_llm(queries_json_str)
        logger.info(f"[{self.name}] Cleaned queries JSON: '{cleaned_queries_json_str[:500]}...'")

        search_queries = []
        try:
            parsed_queries = json.loads(cleaned_queries_json_str)
            if isinstance(parsed_queries, list):
                search_queries = [str(q) for q in parsed_queries if isinstance(q, str)]
            else:
                logger.error(f"[{self.name}] Parsed static asset search queries is not a list: {type(parsed_queries)}")
        except json.JSONDecodeError as e:
            logger.error(f"[{self.name}] Failed to parse static_asset_search_queries_json: {e}. String was: '{cleaned_queries_json_str}'")

        logger.info(f"[{self.name}] Generated {len(search_queries)} search queries to process: {search_queries}")

        player_lookup_dict_json_str = ctx.session.state.get("player_lookup_dict_json", "{}")
        player_lookup_dict = {}
        try:
            player_lookup_dict = json.loads(player_lookup_dict_json_str)
            if not isinstance(player_lookup_dict, dict):
                logger.warning(f"[{self.name}] player_lookup_dict_json parsed into non-dict: {type(player_lookup_dict)}. Defaulting to empty.")
                player_lookup_dict = {}
        except json.JSONDecodeError:
            logger.error(f"[{self.name}] Failed to parse player_lookup_dict_json: '{player_lookup_dict_json_str}'. Defaulting to empty.")

        player_name_to_id_map_lower = {str(name).lower(): str(id_) for id_, name in player_lookup_dict.items() if name}
        logger.info(f"[{self.name}] Player name to ID map (lowercase keys): {player_name_to_id_map_lower}")

        found_assets_list = []
        for original_query_string in search_queries:
            logger.info(f"[{self.name}] Processing query: '{original_query_string}'")
            team_name_for_logo = self._extract_team_name_from_query(original_query_string)
            player_name_for_headshot = self._extract_player_name_from_query(original_query_string)

            if team_name_for_logo:
                logger.info(f"[{self.name}] Identified as LOGO query for team: '{team_name_for_logo}'")
                ctx.session.state["team_name_for_logo_search"] = team_name_for_logo
                async for event in self.logo_searcher_llm.run_async(ctx): yield event

                logo_result_json_str = ctx.session.state.get("logo_search_result_json", "[]")
                cleaned_logo_result_str = _clean_json_string_from_llm(logo_result_json_str)
                logger.info(f"[{self.name}] Logo searcher LlmAgent output: '{cleaned_logo_result_str[:200]}...'")
                try:
                    logo_results = json.loads(cleaned_logo_result_str)
                    if logo_results and isinstance(logo_results, list) and logo_results[0].get("image_uri"):
                        asset_info = logo_results[0]
                        asset_info["search_term_origin"] = original_query_string
                        found_assets_list.append(asset_info)
                        logger.info(f"[{self.name}] Successfully retrieved logo: {asset_info.get('image_uri')}")
                    else:
                        logger.warning(f"[{self.name}] No valid logo found for '{team_name_for_logo}'. Result: {cleaned_logo_result_str}")
                except json.JSONDecodeError as e:
                    logger.error(f"[{self.name}] Failed to parse logo search result: {e}. String was: '{cleaned_logo_result_str}'")

            elif player_name_for_headshot:
                logger.info(f"[{self.name}] Identified as HEADSHOT query for player: '{player_name_for_headshot}'")
                player_id = player_name_to_id_map_lower.get(player_name_for_headshot.lower())
                original_player_name = player_lookup_dict.get(player_id) if player_id else player_name_for_headshot

                if player_id:
                    logger.info(f"[{self.name}] Found Player ID '{player_id}' for '{player_name_for_headshot}'. Original name: '{original_player_name}'")
                    ctx.session.state["player_id_for_headshot_search"] = str(player_id)
                    ctx.session.state["player_name_for_headshot_log"] = original_player_name
                    async for event in self.headshot_retriever_llm.run_async(ctx): yield event

                    headshot_result_json_str = ctx.session.state.get("headshot_uri_result_json", "{}")
                    cleaned_headshot_result_str = _clean_json_string_from_llm(headshot_result_json_str)
                    logger.info(f"[{self.name}] Headshot retriever LlmAgent output: '{cleaned_headshot_result_str[:200]}...'")
                    try:
                        headshot_result = json.loads(cleaned_headshot_result_str)
                        if headshot_result and isinstance(headshot_result, dict) and headshot_result.get("image_uri"):
                            asset_info = headshot_result
                            asset_info["search_term_origin"] = original_query_string
                            found_assets_list.append(asset_info)
                            logger.info(f"[{self.name}] Successfully retrieved headshot: {asset_info.get('image_uri')}")
                        else:
                            logger.warning(f"[{self.name}] No headshot URI found for '{original_player_name}' (ID: {player_id}). Result: {cleaned_headshot_result_str}")
                    except json.JSONDecodeError as e:
                        logger.error(f"[{self.name}] Failed to parse headshot search result: {e}. String was: '{cleaned_headshot_result_str}'")
                else:
                    logger.warning(f"[{self.name}] Player ID NOT FOUND for headshot query: '{player_name_for_headshot}' (searched as '{player_name_for_headshot.lower()}')")
            else:
                logger.warning(f"[{self.name}] Unrecognized query format, skipping: '{original_query_string}'")

        ctx.session.state["current_static_assets_list"] = found_assets_list
        logger.info(f"[{self.name}] Static asset retrieval pipeline finished. Found {len(found_assets_list)} static assets.")
        if found_assets_list:
            logger.info(f"[{self.name}] Details of static assets: {json.dumps(found_assets_list, indent=2)}")
        yield Event(author=self.name, content=types.Content(role="model", parts=[types.Part(text=f"Static asset pipeline complete. Found {len(found_assets_list)} assets.")]))

# --- Tool Collection ---
async def _collect_tools_stack_isolated(
    server_config_dict: AllServerConfigs,
) -> Tuple[Dict[str, Any], contextlib.AsyncExitStack]:
    all_tools: Dict[str, Any] = {}
    exit_stack = contextlib.AsyncExitStack()
    stack_needs_closing = False
    try:
        if not hasattr(server_config_dict, "configs") or not isinstance(
            server_config_dict.configs, dict
        ):
            logger.error("server_config_dict does not have a valid '.configs' dictionary.")
            return {}, exit_stack
        for key, server_params in server_config_dict.configs.items():
            individual_exit_stack: Optional[contextlib.AsyncExitStack] = None
            try:
                tools, individual_exit_stack = await MCPToolset.from_server(
                    connection_params=server_params
                )
                if individual_exit_stack:
                    await exit_stack.enter_async_context(individual_exit_stack)
                    stack_needs_closing = True
                if tools:
                    all_tools[key] = tools
                    logger.info(f"Successfully collected tools for MCP server: {key}")
                else:
                    logging.warning("Connection successful for key '%s', but no tools returned.", key)
            except Exception as e:
                logging.error(f"Failed to connect or get tools for {key}: {e}", exc_info=True)
        if not all_tools:
            logging.warning("No tools were collected from any server for isolated test.")
        return all_tools, exit_stack
    except Exception as e:
        logging.error("Unhandled exception in _collect_tools_stack_isolated: %s", e, exc_info=True)
        if stack_needs_closing:
            await exit_stack.aclose()
        raise

# --- FastAPI Lifespan ---
@asynccontextmanager
async def app_lifespan_isolated(app_instance: FastAPI) -> AsyncGenerator[None, None]:
    global isolated_loaded_mcp_tools
    logging.info("Isolated App Lifespan: Startup initiated.")
    app_instance.state.mcp_tools = {}
    app_instance.state.mcp_tool_exit_stack = None
    try:
        collected_tools, tool_stack = await _collect_tools_stack_isolated(
            server_configs_instance_isolated
        )
        app_instance.state.mcp_tools = collected_tools
        isolated_loaded_mcp_tools = collected_tools
        app_instance.state.mcp_tool_exit_stack = tool_stack
        logging.info(
            "Isolated App Lifespan: MCP Toolset initialized. Tools: %s",
            list(app_instance.state.mcp_tools.keys()),
        )
    except Exception as e:
        logging.error(f"Critical error during isolated MCP Toolset initialization: {e}", exc_info=True)
    yield
    logging.info("Isolated App Lifespan: Shutdown initiated.")
    if app_instance.state.mcp_tool_exit_stack:
        logging.info("Isolated App Lifespan: Closing MCP Toolset connections.")
        try:
            await app_instance.state.mcp_tool_exit_stack.aclose()
            logging.info("Isolated App Lifespan: MCP Toolset connections closed successfully.")
        except Exception as e:
            logging.error(f"Error closing MCP tool connections: {e}", exc_info=True)

app = FastAPI(lifespan=app_lifespan_isolated)

class HardcodedPayloadData(BaseModel):
    current_recap: str
    player_lookup_dict_json: str

DEFAULT_TEST_PAYLOAD = HardcodedPayloadData(
    current_recap="The Los Angeles Dodgers defeated the San Francisco Giants. Mookie Betts had a great game for the Dodgers. The Giants pitcher, Logan Webb, struggled. Freddie Freeman also contributed for the Dodgers.",
    player_lookup_dict_json='{"54321": "Mookie Betts", "67890": "Logan Webb", "11223": "Freddie Freeman"}'
)

@app.post("/test_static_asset_pipeline/{session_id_from_path:str}")
async def test_pipeline_endpoint(
    session_id_from_path: str = Path(..., title="The Session ID for the test run")
) -> Dict[str, Any]:
    logger.info(f"Received request for /test_static_asset_pipeline for session: {session_id_from_path}")

    payload_data = DEFAULT_TEST_PAYLOAD
    session_id_to_use = session_id_from_path

    if not isolated_loaded_mcp_tools.get("static_retriever_mcp") or \
       not isolated_loaded_mcp_tools.get("image_embedding_mcp"):
        logger.error("Required MCP tools (static_retriever_mcp or image_embedding_mcp) are not loaded.")
        raise HTTPException(status_code=503, detail="Essential MCP tools for static assets are not available.")

    try:
        logo_searcher_llm = LlmAgent(
            name="LogoSearcherLlm",
            model=GEMINI_FLASH_MODEL_ID,
            instruction="""Your ONLY task is to call the `image_embedding_mcp.search_similar_images_by_text` tool.
You will be given `team_name_for_logo_search` in session state.
Use these exact parameters for the tool call:
- `query_text`: {session.state.team_name_for_logo_search}
- `top_k`: 1
- `filter_image_type`: "logo"
Your entire output MUST be ONLY the direct, verbatim JSON string that is returned by the tool.
Do NOT add any other text, explanation, or formatting.
""",
            tools=isolated_loaded_mcp_tools.get("image_embedding_mcp", []),
            output_key="logo_search_result_json"
        )

        headshot_retriever_llm = LlmAgent(
            name="HeadshotRetrieverLlm",
            model=GEMINI_FLASH_MODEL_ID,
            instruction="""Your ONLY task is to call the `static_retriever_mcp.get_headshot_uri_if_exists` tool.
You will be given `player_id_for_headshot_search` and `player_name_for_headshot_log` in session state.
Use these exact parameters for the tool call:
- `player_id_str`: {session.state.player_id_for_headshot_search}
- `player_name_for_log`: {session.state.player_name_for_headshot_log}
Your entire output MUST be ONLY the direct, verbatim JSON string that is returned by the tool.
Do NOT add any other text, explanation, or formatting.
""",
            tools=isolated_loaded_mcp_tools.get("static_retriever_mcp", []),
            output_key="headshot_uri_result_json"
        )

        pipeline_agent = StaticAssetPipelineAgent(
            name="TestStaticAssetPipelineAgent",
            entity_extractor=entity_extractor_for_assets_agent,
            static_asset_query_generator=static_asset_query_generator_for_assets_agent,
            logo_searcher_llm=logo_searcher_llm,
            headshot_retriever_llm=headshot_retriever_llm
        )

        # --- Refined Session Handling ---
        session: Optional[Session] = None
        try:
            # Ensure session_id_to_use is treated as the unique key for user_id and session_id parts
            session = isolated_session_service.get_session(
                app_name=ISOLATED_APP_NAME, user_id=session_id_to_use, session_id=session_id_to_use
            )
            if session is not None:
                logger.info(f"Reusing existing session: {session_id_to_use}. Session state keys: {list(session.state.keys()) if session.state else 'empty'}")
            # If session is None here, it means the key existed but mapped to None (highly unlikely for InMemorySessionService)
            # or there's a deeper issue. The `if session is None:` below will catch it.

        except KeyError:
            logger.info(f"Session not found with key ({ISOLATED_APP_NAME}, user_id='{session_id_to_use}', session_id='{session_id_to_use}'). Will create a new one.")
            # `session` remains None, which is fine as it triggers creation below.
            pass

        if session is None: # Create if not found by get_session or if get_session (unexpectedly) returned None
            logger.info(f"Attempting to create new session for key ({ISOLATED_APP_NAME}, user_id='{session_id_to_use}', session_id='{session_id_to_use}')")
            session = isolated_session_service.create_session(
                app_name=ISOLATED_APP_NAME, user_id=session_id_to_use, session_id=session_id_to_use, state={}
            )
            logger.info(f"Created new session: {session_id_to_use}. Initial state: {session.state if session else 'ERROR - Session still None after create'}")

        if session is None: # Should be absolutely impossible to reach here if create_session works
            logger.error(f"CRITICAL FAILURE: Session object is None for {session_id_to_use} even after create attempt.")
            raise HTTPException(status_code=500, detail="Internal server error: Failed to initialize session.")
        # --- End of Refined Session Handling ---


        session.state["current_recap"] = payload_data.current_recap
        session.state["player_lookup_dict_json"] = payload_data.player_lookup_dict_json
        session.state["static_asset_search_queries_json"] = "[]"
        session.state["logo_search_result_json"] = "[]"
        session.state["headshot_uri_result_json"] = "{}"

        runner = Runner(
            app_name=ISOLATED_APP_NAME,
            agent=pipeline_agent,
            artifact_service=isolated_artifacts_service,
            session_service=isolated_session_service,
        )
        dummy_content = types.Content(role="user", parts=[types.Part(text="trigger run")])
        logger.info(f"Running StaticAssetPipelineAgent for session: {session_id_to_use}")
        async for event in runner.run_async(
            session_id=session_id_to_use, user_id=session_id_to_use, new_message=dummy_content
        ):
            if event.content and event.content.parts:
                logger.info(f"Event from {event.author}: {event.content.parts[0].text}")
            else:
                logger.info(f"Event from {event.author} (no direct content parts)")
        logger.info(f"StaticAssetPipelineAgent run completed for session: {session_id_to_use}")

        # Re-fetch session state as runner operations might modify it in its own context.
        final_session_object = isolated_session_service.get_session(
             app_name=ISOLATED_APP_NAME, user_id=session_id_to_use, session_id=session_id_to_use
        )
        if final_session_object is None:
            logger.error(f"Failed to re-fetch session state for {session_id_to_use} after agent run.")
            raise HTTPException(status_code=500, detail="Failed to retrieve session state post-run.")

        final_session_state = final_session_object.state
        retrieved_assets = final_session_state.get("current_static_assets_list", [])

        return {
            "session_id": session_id_to_use,
            "message": "StaticAssetPipelineAgent executed with hardcoded payload.",
            "retrieved_assets": retrieved_assets,
            "full_final_state_sample": {
                 "extracted_entities_json": final_session_state.get("extracted_entities_json"),
                 "static_asset_search_queries_json": final_session_state.get("static_asset_search_queries_json"),
                 "player_lookup_dict_original_input": payload_data.player_lookup_dict_json,
                 "player_lookup_dict_from_state_for_agent": final_session_state.get("player_lookup_dict_json")
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during /test_static_asset_pipeline: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting isolated FastAPI app for StaticAssetPipelineAgent testing on port 8001.")
    logger.info("Ensure that the MCP server scripts are running or can be started by this app.")
    logger.info("MCP Server paths used:")
    logger.info(f"  Static Retriever: {static_retriever_server_params_isolated.args}")
    logger.info(f"  Image Embedding: {image_embedding_server_params_isolated.args}")
    uvicorn.run(app, host="0.0.0.0", port=8001)