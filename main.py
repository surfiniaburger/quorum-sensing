"""
Main function to run FastAPI server.
"""

import asyncio
import contextlib
from contextlib import asynccontextmanager
import json
import hashlib
from typing import Any, Dict, List, Optional, Tuple, AsyncGenerator
from typing_extensions import override
from fastapi import HTTPException
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from google.adk.agents import LlmAgent, BaseAgent, LoopAgent, SequentialAgent # Added BaseAgent, LoopAgent, SequentialAgent
from google.adk.agents.invocation_context import InvocationContext # Added
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters
from google.genai import types
from google.adk.events import Event # Added
from pydantic import BaseModel, Field
from starlette.websockets import WebSocketDisconnect
from voice_agent import router as voice_agent_router
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__) # Use this for consistency

# --- Configuration & Global Setup ---
load_dotenv()

APP_NAME = "ADK_MCP_App_Updated" # Changed to avoid potential conflicts if old app is running
MODEL_ID = "gemini-2.0-flash" # Updated to a generally available model, ensure you have access
GEMINI_PRO_MODEL_ID = "gemini-2.5-flash-preview-04-17" # For potentially more complex tasks like generation/revision

STATIC_DIR = "static"

# Initialize services (globally or via dependency injection)
session_service = InMemorySessionService()
artifacts_service = InMemoryArtifactService()

# Global variable to hold loaded MCP tools after lifespan startup
loaded_mcp_tools_global: Dict[str, Any] = {}

class AllServerConfigs(BaseModel):
    configs: Dict[str, StdioServerParameters]

# --- Server Parameter Definitions ---
weather_server_params = StdioServerParameters(
    command="python",
    args=["./mcp_server/weather_server.py"],
)
ct_server_params = StdioServerParameters(
    command="python",
    args=["./mcp_server/cocktail.py"],
)
bnb_server_params = StdioServerParameters(
    command="npx", args=["-y", "@openbnb/mcp-server-airbnb", "--ignore-robots-txt"]
)
mlb_stats_server_params = StdioServerParameters(
    command="python",
    args=["./mcp_server/mlb_stats_server.py"],
)
web_search_server_params = StdioServerParameters( # NEW
    command="python",
    args=["./mcp_server/web_search_server.py"],
)
bq_vector_search_server_params = StdioServerParameters( # NEW
    command="python",
    args=["./mcp_server/bq_vector_search_server.py"],
)
visual_asset_server_params = StdioServerParameters( # Your existing Imagen/Cloudflare server
    command="python",
    args=["./mcp_server/visual_asset_server.py"], # Path to your image generation server
)
static_retriever_server_params = StdioServerParameters( # NEW - for headshots
    command="python",
    args=["./mcp_server/static_asset_retriever_mcp_server.py"],
)
image_embedding_server_params = StdioServerParameters( # Your existing for vector search (logos)
    command="python",
    args=["./mcp_server/image_embedding_server.py"],
)


server_configs_instance = AllServerConfigs(
    configs={
        "weather": weather_server_params,
        "bnb": bnb_server_params,
        "ct": ct_server_params,
        "mlb": mlb_stats_server_params,
        "web_search": web_search_server_params,         # NEW
        "bq_search": bq_vector_search_server_params, # NEW
        "visual_assets": visual_asset_server_params, # MCP for Imagen/Cloudflare generation
        "static_retriever_mcp": static_retriever_server_params, # MCP for GCS headshot check
        "image_embedding_mcp": image_embedding_server_params, # MCP for logo vector search
    }
)

# --- Agent Instructions ---
ROOT_AGENT_INSTRUCTION = """
**Role:** You are a Virtual Assistant acting as a Request Router.
**Primary Goal:** Analyze user requests and route them to the correct specialist sub-agent.
**Capabilities & Routing:**
* **Greetings:** If the user greets you, respond warmly and directly.
* **Cocktails:** Route requests about cocktails, drinks, recipes, or ingredients to `cocktail_assistant`.
* **Booking & Weather:** Route requests about booking accommodations or weather to `booking_assistant`.
* **MLB Information (General):** Route general requests concerning Major League Baseball (MLB) stats, scores, schedules, rosters, standings to the `mlb_assistant`.
    The `mlb_assistant` will handle obtaining any necessary IDs (like `game_pk`, `player_id`, `team_id`) if not provided by the user for these general queries.
* **MLB Game Recap:** If the user specifically asks for a "game recap", "recap of the game", "game summary" or similar, and a specific game can be identified (e.g., "recap of yesterday's Yankees game" or "recap for game PK 12345"), route the request to the `game_recap_assistant`.
    - If a `game_pk` is mentioned or easily derivable from the query (e.g. from a team name and date like "yesterday's Yankees game"), include it or the identifying information in the routing.
    - If the game for the recap is unclear, you can first delegate to `mlb_assistant` to help identify the `game_pk`, and then if `game_pk` is found, the user might be prompted to ask for the recap again, or you could try re-routing. (Simpler: for now, assume if routed to `game_recap_assistant`, the query contains enough info to derive game_pk, or `game_recap_assistant` will handle clarification if needed).
* **Out-of-Scope:** If the request is unrelated, state directly that you cannot assist.
**Key Directives:**
* **Delegate Immediately:** Once a suitable sub-agent is identified, route the request.
* **Do Not Answer Delegated Topics:** You must **not** attempt to answer questions for delegated topics yourself.
* **Formatting:** Format your final response using Markdown.
* **Game Recap Clarification:** If a user asks for a game recap but doesn't specify which game, ask them to specify the game (e.g., "Which game would you like a recap for? Please provide the teams and date, or the game ID if you know it.") before attempting to route to `game_recap_assistant`. If they provide details, then route.
"""


###########################################################################################################################################################################################################
###########################################################################################################################################################################################################
# --- VisualAssetWorkflowAgent Definition ---
# This agent orchestrates the entire visual asset workflow, including static asset retrieval,
# In main.py, after LlmAgent definitions for visuals

# In main.py

class VisualAssetWorkflowAgent(BaseAgent):
    model_config = {"arbitrary_types_allowed": True}
    entity_extractor: LlmAgent
    static_asset_query_generator: LlmAgent
    static_asset_retriever: LlmAgent
    generated_visual_prompts_generator: LlmAgent
    visual_generator_mcp_caller: LlmAgent
    visual_critic: LlmAgent
    new_visual_prompts_creator: LlmAgent
    max_visual_refinement_loops: int

    def __init__(self, name: str, entity_extractor: LlmAgent, static_asset_query_generator: LlmAgent,
                 static_asset_retriever: LlmAgent, generated_visual_prompts_generator: LlmAgent,
                 visual_generator_mcp_caller: LlmAgent, visual_critic: LlmAgent,
                 new_visual_prompts_creator: LlmAgent, max_visual_refinement_loops: int = 1):
        sub_agents_list_for_framework = [
            entity_extractor, static_asset_query_generator, static_asset_retriever,
            generated_visual_prompts_generator, visual_generator_mcp_caller,
            visual_critic, new_visual_prompts_creator,
        ]
        super().__init__(
            name=name, entity_extractor=entity_extractor,
            static_asset_query_generator=static_asset_query_generator,
            static_asset_retriever=static_asset_retriever,
            generated_visual_prompts_generator=generated_visual_prompts_generator,
            visual_generator_mcp_caller=visual_generator_mcp_caller,
            visual_critic=visual_critic, new_visual_prompts_creator=new_visual_prompts_creator,
            max_visual_refinement_loops=max_visual_refinement_loops,
            sub_agents=sub_agents_list_for_framework
        )

    def _clean_json_string_from_llm(self, raw_json_string: Optional[str], default_if_empty: str = "[]") -> str:
        if not raw_json_string: return default_if_empty
        clean_string = raw_json_string.strip()
        if clean_string.startswith("```json"):
            clean_string = clean_string[7:]
            if clean_string.endswith("```"): clean_string = clean_string[:-3]
        elif clean_string.startswith("```"):
            clean_string = clean_string[3:]
            if clean_string.endswith("```"): clean_string = clean_string[:-3]
        return clean_string.strip() if clean_string.strip() else default_if_empty

    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        logger.info(f"[{self.name}] Starting visual asset workflow.")
        if not ctx.session.state.get("current_recap"):
            logger.warning(f"[{self.name}] 'current_recap' missing. Skipping visual workflow.")
            ctx.session.state["all_visual_assets_list"] = []
            return

        # --- 1. Static Asset Retrieval Path ---
        logger.info(f"[{self.name}] Running EntityExtractorForAssets...")
        async for event in self.entity_extractor.run_async(ctx): yield event
        
        # Ensure 'extracted_entities_json' is clean before passing to next agent
        raw_entities = ctx.session.state.get("extracted_entities_json", "{}")
        ctx.session.state["extracted_entities_json"] = self._clean_json_string_from_llm(raw_entities, default_if_empty='{"players":[], "teams":[]}')

        logger.info(f"[{self.name}] Running StaticAssetQueryGenerator...")
        async for event in self.static_asset_query_generator.run_async(ctx): yield event

        # Ensure player_lookup_dict_json is present (should be set by GameRecapAgent)
        if "player_lookup_dict_json" not in ctx.session.state:
            ctx.session.state["player_lookup_dict_json"] = "{}"
            logger.warning(f"[{self.name}] 'player_lookup_dict_json' was not in session state. Defaulted to empty. Headshot retrieval may be affected.")

        logger.info(f"[{self.name}] Running StaticAssetRetriever...")
        async for event in self.static_asset_retriever.run_async(ctx): yield event
        
        retrieved_static_assets_json_clean = self._clean_json_string_from_llm(ctx.session.state.get("retrieved_static_assets_json", "[]"))
        try:
            current_static_assets_list = json.loads(retrieved_static_assets_json_clean)
            if not isinstance(current_static_assets_list, list): current_static_assets_list = []
        except json.JSONDecodeError: current_static_assets_list = []
        logger.info(f"[{self.name}] Retrieved {len(current_static_assets_list)} static assets.")

        # --- 2. Iterative Generative Visuals Workflow ---
        logger.info(f"[{self.name}] Running GeneratedVisualPromptsGenerator (for initial prompts)...")
        async for event in self.generated_visual_prompts_generator.run_async(ctx): yield event
        
        # Initial prompts are set by generated_visual_prompts_generator into 'visual_generation_prompts_json'
        # This will be the starting point for the loop's prompt source.
        current_prompts_json_for_next_iteration = self._clean_json_string_from_llm(
            ctx.session.state.get("visual_generation_prompts_json", "[]")
        )
        logger.info(f"[{self.name}] Initial cleaned prompts JSON for visual gen: {current_prompts_json_for_next_iteration}")


        all_generated_assets_details = []

        for i in range(self.max_visual_refinement_loops + 1):
            iteration_label = f"Iteration {i+1}/{self.max_visual_refinement_loops + 1}"
            logger.info(f"[{self.name}] Visual generation/refinement {iteration_label}")
            
            # Use the prompts prepared for this iteration
            prompts_to_use_this_iteration_str = current_prompts_json_for_next_iteration
            
            current_prompts_list_for_iter = []
            try:
                parsed_list = json.loads(prompts_to_use_this_iteration_str)
                if isinstance(parsed_list, list) and parsed_list: # Ensure it's a non-empty list
                    current_prompts_list_for_iter = parsed_list
                else:
                    log_msg = f"[{self.name}] No valid visual prompts for {iteration_label}."
                    if i == 0: logger.warning(log_msg + " Ending visual generation. Prompt string was: '{prompts_to_use_this_iteration_str}'")
                    else: logger.info(log_msg + " Ending visual refinement.")
                    break 
            except json.JSONDecodeError as e:
                logger.error(f"[{self.name}] Invalid JSON for prompts in {iteration_label}. Error: {e}. JSON: '{prompts_to_use_this_iteration_str}'. Stopping.")
                break
            
            # Set the exact key the VisualGeneratorMCPCaller LlmAgent expects
            ctx.session.state['visual_generation_prompts_json_for_tool'] = prompts_to_use_this_iteration_str
            current_game_pk = ctx.session.state.get("game_pk", "unknown_game")
            ctx.session.state['game_pk_str_for_tool'] = str(current_game_pk) # LLM reads this
            logger.info(f"[{self.name}] {iteration_label}: Set 'visual_generation_prompts_json_for_tool' to: {prompts_to_use_this_iteration_str}")
            logger.info(f"[{self.name}] {iteration_label}: Set 'game_pk_str_for_tool' to: {str(current_game_pk)}")
            
            logger.info(f"[{self.name}] {iteration_label}: Calling VisualGeneratorMCPCaller with {len(current_prompts_list_for_iter)} prompts.")
            async for event in self.visual_generator_mcp_caller.run_async(ctx): yield event
            
            generated_uris_this_iter_json_raw = ctx.session.state.get("generated_visual_assets_uris_json", "[]")
            # It's crucial that visual_generator_mcp_caller_agent's output (which becomes generated_visual_assets_uris_json)
            # is a JSON string from the MCP tool, even if it's an error JSON like '{"error": "..."}' or an empty list '[]'
            generated_uris_this_iter_json_clean = self._clean_json_string_from_llm(generated_uris_this_iter_json_raw, default_if_empty="[]")
            logger.info(f"[{self.name}] {iteration_label}: Raw output from VisualGeneratorMCPCaller: '{generated_uris_this_iter_json_raw}'")
            logger.info(f"[{self.name}] {iteration_label}: Cleaned output from VisualGeneratorMCPCaller: '{generated_uris_this_iter_json_clean}'")

            generated_uris_this_iter = []
            tool_had_error = False
            try:
                parsed_uris_result = json.loads(generated_uris_this_iter_json_clean)
                if isinstance(parsed_uris_result, list):
                    generated_uris_this_iter = [uri for uri in parsed_uris_result if isinstance(uri, str) and uri.startswith("gs://")]
                    logger.info(f"[{self.name}] {iteration_label}: Successfully parsed {len(generated_uris_this_iter)} URIs.")
                elif isinstance(parsed_uris_result, dict) and parsed_uris_result.get("error"):
                    logger.error(f"[{self.name}] {iteration_label}: VisualGeneratorMCPCaller tool explicitly returned an error: {parsed_uris_result['error']}")
                    tool_had_error = True # Mark that the tool itself reported an error
                else:
                    logger.warning(f"[{self.name}] {iteration_label}: VisualGeneratorMCPCaller output was not a list of URIs or an error dict: {generated_uris_this_iter_json_clean}")
            except json.JSONDecodeError as e:
                logger.error(f"[{self.name}] {iteration_label}: Failed to parse JSON output from VisualGeneratorMCPCaller. Error: {e}. Cleaned JSON: '{generated_uris_this_iter_json_clean}'")
                tool_had_error = True # Treat parse error as a tool failure for this iteration

            # Prepare assets for the critic, even if generation failed for some/all
            assets_for_critique_this_iteration = []
            for idx, prompt_text in enumerate(current_prompts_list_for_iter): # Use the parsed list of prompts
                asset_uri = generated_uris_this_iter[idx] if idx < len(generated_uris_this_iter) else None
                assets_for_critique_this_iteration.append({"prompt_origin": prompt_text, "image_uri": asset_uri, "type": "generated_image"})
                if asset_uri: # Only add successfully generated assets to the main list
                    all_generated_assets_details.append({"prompt_origin": prompt_text, "image_uri": asset_uri, "type": "generated_image", "iteration": i + 1})
            
            ctx.session.state["assets_for_critique_json"] = json.dumps(assets_for_critique_this_iteration)
            ctx.session.state["prompts_used_for_critique_json"] = prompts_to_use_this_iteration_str

            if i >= self.max_visual_refinement_loops:
                logger.info(f"[{self.name}] Reached max visual refinement loops ({self.max_visual_refinement_loops}). No further critique.")
                break
            
            # If the tool itself had an error (e.g., Imagen quota, invalid input to MCP tool),
            # the critique might not be useful, or it might suggest retrying.
            # For now, we proceed to critique even if tool_had_error is true, the critic can see null URIs.

            logger.info(f"[{self.name}] {iteration_label}: Running VisualCritic...")
            async for event in self.visual_critic.run_async(ctx): yield event
            
            critique_text = ctx.session.state.get("visual_critique_text", "")
            logger.info(f"[{self.name}] {iteration_label} Visual Critique: {critique_text[:150]}...")
            if "sufficient" in critique_text.lower():
                logger.info(f"[{self.name}] Visuals deemed sufficient by critic. Ending visual refinement.")
                break

            logger.info(f"[{self.name}] {iteration_label}: Running NewVisualPromptsFromCritique...")
            async for event in self.new_visual_prompts_creator.run_async(ctx): yield event
            
            new_prompts_json_raw = ctx.session.state.get("new_visual_generation_prompts_json", "[]")
            current_prompts_json_for_next_iteration = self._clean_json_string_from_llm(new_prompts_json_raw) # Prepare for next loop
            logger.info(f"[{self.name}] {iteration_label}: New prompts for next iter: {current_prompts_json_for_next_iteration}")

            try:
                new_prompts_list = json.loads(current_prompts_json_for_next_iteration)
                if not isinstance(new_prompts_list, list) or not new_prompts_list:
                    logger.info(f"[{self.name}] Critique yielded no new prompts. Ending visual refinement loop after {iteration_label}.")
                    break
            except json.JSONDecodeError:
                logger.error(f"[{self.name}] Invalid JSON for new prompts: '{current_prompts_json_for_next_iteration}'. Ending.")
                break
        
        # ... (rest of the _run_async_impl: combining static and generated assets) ...
        final_generated_visuals_dict = {asset["image_uri"]: asset for asset in all_generated_assets_details if asset.get("image_uri")}
        all_visual_assets_list = current_static_assets_list + list(final_generated_visuals_dict.values())
        ctx.session.state["all_visual_assets_list"] = all_visual_assets_list
        logger.info(f"[{self.name}] Visual asset workflow finished. Total unique assets: {len(all_visual_assets_list)}")
        logger.info(f"[{self.name}] Generated visuals workflow finished. Total assets: {len(final_generated_visuals_dict.values())}")


#################################################################################################################
#################################################################################################################
# --- GameRecapAgent Definition (Patterned after StoryFlowAgent) ---

class GameRecapAgent(BaseAgent):
    """
    Custom agent for generating and refining an MLB game recap.
    Orchestrates fetching data, generation, critique, enrichment, and revision.
    """
    model_config = {"arbitrary_types_allowed": True}

    # Define sub-agents that will be passed during initialization
    initial_recap_generator: LlmAgent
    recap_critic: LlmAgent
    critique_processor: LlmAgent # Processes critique, stores it, generates search queries, runs searches
    recap_reviser: LlmAgent
    grammar_check: LlmAgent
    tone_check: LlmAgent

    refinement_loop: LoopAgent
    post_processing_sequence: SequentialAgent
    visual_asset_workflow: VisualAssetWorkflowAgent # The new visual agent


    def __init__(
        self,
        name: str,

        initial_recap_generator: LlmAgent,
        recap_critic: LlmAgent,
        critique_processor: LlmAgent,
        recap_reviser: LlmAgent,
        grammar_check: LlmAgent,
        tone_check: LlmAgent,
        visual_asset_workflow: VisualAssetWorkflowAgent, # Pass it in
    ):
        # Create internal composite agents
        # The loop will run: Critic -> CritiqueProcessor -> Reviser
        refinement_loop = LoopAgent(
            name="RecapRefinementLoop",
            sub_agents=[recap_critic, critique_processor, recap_reviser],
            max_iterations=1 # Configurable number of refinement cycles
        )
        post_processing_sequence = SequentialAgent(
            name="RecapPostProcessing",
            sub_agents=[grammar_check, tone_check]
        )

        # Define the sub_agents list for the framework
        # These are the agents that GameRecapAgent directly invokes in its _run_async_impl
        sub_agents_list = [
            initial_recap_generator,
            refinement_loop,
            post_processing_sequence,
            visual_asset_workflow,
        ]

        super().__init__(
            name=name,
       
            initial_recap_generator=initial_recap_generator,
            recap_critic=recap_critic,
            critique_processor=critique_processor,
            recap_reviser=recap_reviser,
            grammar_check=grammar_check,
            tone_check=tone_check,
            refinement_loop=refinement_loop,
            post_processing_sequence=post_processing_sequence,
            sub_agents=sub_agents_list,
            visual_asset_workflow=visual_asset_workflow, 
        )

# In main.py, within GameRecapAgent class

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        logger.info(f"[{self.name}] Starting full game recap and visual workflow.")

        # --- 0. Pre-requisites (user_query, game_pk, player_lookup_dict_json) ---
        user_query = ctx.session.state.get("user_query")
        if not user_query:
            if ctx.user_content and ctx.user_content.parts and ctx.user_content.parts[0].text:
                # Simplistic assumption: user_content.parts[0].text is the query
                # Robust parsing might be needed if user_content is structured JSON
                try:
                    data = json.loads(ctx.user_content.parts[0].text)
                    if isinstance(data, dict) and "message" in data:
                        user_query = data["message"]
                    else:
                        user_query = ctx.user_content.parts[0].text
                except json.JSONDecodeError:
                    user_query = ctx.user_content.parts[0].text
                ctx.session.state["user_query"] = user_query
            else:
                user_query = "generic recap request"
                ctx.session.state["user_query"] = user_query
            logger.info(f"[{self.name}] User query set to: '{user_query}'")

        # Ensure game_pk is in state (InitialRecapGenerator might set this if not present)
        if "game_pk" not in ctx.session.state:
            logger.warning(f"[{self.name}] game_pk not in session state before InitialRecapGenerator. It will attempt to derive it.")
        
        # Ensure player_lookup_dict_json for VisualAssetWorkflowAgent
        if "player_lookup_dict_json" not in ctx.session.state:
            logger.info(f"[{self.name}] 'player_lookup_dict_json' not in state. Attempting to derive from 'player_id_to_name_map' (if set by mlb_assistant).")
            player_id_map = ctx.session.state.get("player_id_to_name_map")
            if player_id_map and isinstance(player_id_map, dict):
                ctx.session.state["player_lookup_dict_json"] = json.dumps(player_id_map)
                logger.info(f"[{self.name}] Populated 'player_lookup_dict_json' from 'player_id_to_name_map'.")
            else:
                logger.warning(f"[{self.name}] Could not populate 'player_lookup_dict_json'. Using empty default. Headshot retrieval might be limited.")
                ctx.session.state["player_lookup_dict_json"] = "{}"


        # --- 1. Dialogue Generation & Refinement ---
        logger.info(f"[{self.name}] Running InitialRecapGenerator (Dialogue)...")
        async for event in self.initial_recap_generator.run_async(ctx): yield event
        
        dialogue_after_initial_gen = ctx.session.state.get("current_recap", "")
        agent_should_exit = ctx.session.state.get("agent_should_exit_flag", False) # Check for an explicit exit flag

        if agent_should_exit or not dialogue_after_initial_gen or "game is not final" in dialogue_after_initial_gen.lower():
            message = dialogue_after_initial_gen or "Recap generation stopped early due to game status or other conditions."
            logger.warning(f"[{self.name}] Initial recap phase indicated exit. Finalizing with message: {message}")
            yield Event(
                author=self.name,
                content=types.Content(role="model", parts=[types.Part(text=message)]),
                # No invocation_context here
            )
            return

        logger.info(f"[{self.name}] Running Dialogue RefinementLoop (self.refinement_loop)...")
        async for event in self.refinement_loop.run_async(ctx): yield event
        
        logger.info(f"[{self.name}] Running Dialogue PostProcessing Sequence (self.post_processing_sequence)...")
        async for event in self.post_processing_sequence.run_async(ctx): yield event

        final_dialogue_recap = ctx.session.state.get("current_recap", "")
        if not final_dialogue_recap:
            logger.error(f"[{self.name}] Dialogue recap is empty after refinement. Aborting.")
            yield Event(
                author=self.name,
                content=types.Content(role="model", parts=[types.Part(text="Failed to produce dialogue recap after refinement.")])
                # No invocation_context here
            )
            return
        logger.info(f"[{self.name}] Dialogue workflow finished. Final dialogue (first 100 chars): {final_dialogue_recap[:100]}...")

        # --- 2. Visual Asset Workflow ---
        logger.info(f"[{self.name}] Running VisualAssetWorkflow (self.visual_asset_workflow)...")
        async for event in self.visual_asset_workflow.run_async(ctx): yield event
        
        all_visual_assets = ctx.session.state.get("all_visual_assets_list", [])
        logger.info(f"[{self.name}] VisualAssetWorkflow finished. Found/generated {len(all_visual_assets)} visual assets (available in session state).")

        # --- 3. Final Output Event for GameRecapAgent ---
        logger.info(f"[{self.name}] Yielding final dialogue recap as the main textual output.")
        final_event_for_dialogue = Event(
            author=self.name,
            content=types.Content(role="model", parts=[types.Part(text=final_dialogue_recap)]),
            # No invocation_context here. Framework handles it.
            # ADK will automatically populate id, timestamp, and associate with the current invocation.
        )
        yield final_event_for_dialogue

        logger.info(f"[{self.name}] === GameRecapAgent processing complete. Yielded final dialogue. Visuals in state. ===")
############################################################################################################################################
############################################################################################################################################


# --- Tool Collection ---
async def _collect_tools_stack(
    server_config_dict: AllServerConfigs,
) -> Tuple[Dict[str, Any], contextlib.AsyncExitStack]:
    all_tools: Dict[str, Any] = {}
    exit_stack = contextlib.AsyncExitStack()
    stack_needs_closing = False
    try:
        if not hasattr(server_config_dict, "configs") or not isinstance(
            server_config_dict.configs, dict
        ):
            logging.error("server_config_dict does not have a valid '.configs' dictionary.")
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
            except FileNotFoundError as file_error:
                logging.error("Command or script not found for key '%s': %s", key, file_error)
            except ConnectionRefusedError as conn_refused:
                logging.error("Connection refused for key '%s': %s", key, conn_refused)
            except Exception as e:
                logging.error(f"Failed to connect or get tools for {key}: {e}", exc_info=True)


        if not all_tools:
            logging.warning("No tools were collected from any server.")

        # Ensure all expected keys exist, even if empty
        expected_keys = ["weather", "bnb", "ct", "mlb", "web_search", "bq_search"]
        for k in expected_keys:
            if k not in all_tools:
                logging.info("Tools for key '%s' were not collected. Ensuring key exists with empty list.", k)
                all_tools[k] = []
        return all_tools, exit_stack
    except Exception as e:
        logging.error("Unhandled exception in _collect_tools_stack: %s", e, exc_info=True)
        if stack_needs_closing:
            await exit_stack.aclose()
        raise


# --- Agent Creation ---
async def create_agent_with_preloaded_tools(
    loaded_mcp_tools: Dict[str, Any],
) -> LlmAgent:
    booking_tools = loaded_mcp_tools.get("bnb", [])
    weather_tools = loaded_mcp_tools.get("weather", [])
    combined_booking_tools = list(booking_tools) + list(weather_tools) # Ensure they are lists
    ct_tools = loaded_mcp_tools.get("ct", [])
    mlb_tools = loaded_mcp_tools.get("mlb", [])
    web_search_tools = loaded_mcp_tools.get("web_search", [])
    bq_search_tools = loaded_mcp_tools.get("bq_search", [])

    # Tools for GameRecapAgent and its sub-agents
    # These LlmAgents will need specific tools from the broader set.
    game_recap_tool_list = list(mlb_tools) + list(web_search_tools) + list(bq_search_tools)


    booking_agent = LlmAgent(
        model=MODEL_ID,
        name="booking_assistant",
        instruction="""Use booking_tools to handle inquiries related to
        booking accommodations and weather information.
        Format your response using Markdown.
        If you don't know how to help, call "agent_exit".""",
        tools=combined_booking_tools,
    )

    cocktail_agent = LlmAgent(
        model=MODEL_ID,
        name="cocktail_assistant",
        instruction="""Use ct_tools to handle all inquiries related to cocktails.
        Format your response using Markdown.
        If you don't know how to help, call "agent_exit".""",
        tools=ct_tools,
    )

    mlb_assistant = LlmAgent(
         model=MODEL_ID,
         name="mlb_assistant",
         instruction="""You are an MLB Stats assistant.
         Use your tools (e.g., `mlb.get_live_game_score`, `mlb.get_team_schedule`) to answer questions about Major League Baseball.
         If the user does not provide all necessary IDs (like `game_pk`, `player_id`, `team_id`), you MUST ask for them before calling the tool.
         If a user asks for a game recap, and you identify a game_pk, you should state that you can provide stats and scores, but for a full recap, they might want to ask the main assistant to route them to the recap specialist. Or, provide the game_pk and use agent_exit.
         Format your responses clearly using Markdown.
         If you cannot help, use "agent_exit".""",
         tools=mlb_tools,
    )

    # --- Sub-Agents for GameRecapAgent ---
    # Ensure `game_pk` is consistently available in session state for these agents.
    # `user_query` or a `task_description` should also be in session state.
    
    # Note: Tool names used by LlmAgent instructions should match how ADK makes them available
    # e.g., if bq_search_tools contains a tool `search_past_critiques` from server "bq_search",
    # it might be callable as `bq_search.search_past_critiques` in the prompt.

    initial_recap_generator_agent = LlmAgent(
        name="InitialRecapGenerator",
        model=GEMINI_PRO_MODEL_ID,
        instruction="""
You are an expert sports journalist tasked with generating an initial MLB game recap in a **two-host dialogue script format**. Your goal is to create a compelling narrative of the game, not just a list of events,  presented as a conversation.

Session State Expectations:
- `game_pk` (e.g., 717527): The unique ID for the game.
- `user_query`: The user's original request (e.g., "recap of Pirates last game").
- `parsed_game_pk_from_query` (Optional): A game_pk parsed directly from the user's query. Prioritize this.
- `pre_game_context_notes` (Optional): Web search findings about rivalry, storylines, etc.
- `past_critiques_feedback` (Optional): General feedback from past similar tasks to guide style and tone.

Your Multi-Step Process:

1.  **Game Identification (Critical):**
    *   If `game_pk` is not in session state: Analyze `user_query`.
    *   If the query implies a "latest" game (e.g., "Brewers last game"), use `mlb.get_team_schedule` (e.g., `days_range=-7`) to find the most recent *final* game. Extract its `game_pk`. If ambiguity, ask for clarification.
    *   Announce the identified game: "Okay, I've identified the game: Team A vs Team B on YYYY-MM-DD, Game PK: [game_pk]." Update `session.state.game_pk`.
    *   If no specific game can be confidently identified, state this and request clarification. Do NOT invent a `game_pk`. Use `agent_exit`.

2.  **Gather Core Game Data (if `game_pk` is now known):**
    *   Call `mlb.get_live_game_score` for `game_pk`.
    *   Call `mlb.get_game_play_by_play_summary` for `game_pk` (get enough plays, e.g., 15-20, to understand key moments).
    *   Call `mlb_stats.get_game_boxscore_and_details` for `game_pk`.
    *   Parse the data: game status, final score, winning/losing pitchers, key offensive performers (multi-hit, RBIs, HRs), inning-by-inning scores.
    *   **If Game Status is "Scheduled" or not "Final"**: Your output MUST state that a full recap is not yet available as the game is not final. Then use `agent_exit`.

3.  **Synthesize Initial Dialogue Script (Only if Game is "Final"):**
    *   **Dialogue Format:**
        *   The entire output MUST be a conversation between two hosts (e.g., Host A, Host B).
        *   **Strict Alternation:** Each line of the script MUST represent one host speaking, alternating strictly.
        *   **NO Speaker Labels:** CRITICAL: Do NOT include speaker labels like "Host 1:", "Host 2:", or any character names. Just write the raw dialogue line for each speaker's turn.
    *   **Storytelling First:** Your primary goal is to tell the story of the game through this dialogue.
        *   Identify a potential "story of the game" (e.g., a pitcher's duel, an offensive breakout, a key player's heroics, a specific turning point).
        *   The dialogue should start with an engaging lead, perhaps one host setting the scene and the other reacting or adding initial thoughts, summarizing the game's outcome and main storyline.
    *   **Pitching Narrative:** The hosts should discuss the performance of the starting pitchers, especially the winner and loser.
    *   **Offensive Highlights & Progression:**
        *   The dialogue should cover how the scoring unfolded, focusing on the most impactful plays.
        *   Hosts should name the players involved in these key offensive moments and discuss their contributions.
    *   **Integrate Context:**
        *   If `pre_game_context_notes` are available, one host might bring it up, and the other can elaborate or connect it to game events.
    *   **Guidance from Past Critiques:** Use `past_critiques_feedback` (if available) for general guidance on narrative structure, tone, and conversational style.
    *   **Language:** Use vivid, active language. The dialogue should sound natural and engaging. Avoid one host just listing stats for the other to react to; make it a genuine discussion.
    *   **Acknowledge Limitations:** If specific details are unavailable, one host might pose it as a question the other can't fully answer, or they might acknowledge the gap.

Output ONLY the generated recap text. Do not add conversational fluff like "Here is the recap..." or "I have gathered the data...".
        """,
        tools=[ # Ensure all necessary tools are listed
            tool for toolset_name in ["bq_search", "mlb", "web_search"] 
            for tool in loaded_mcp_tools.get(toolset_name, [])
        ],
        output_key="current_recap",
    )

    recap_critic_agent = LlmAgent(
        name="RecapCritic",
        model=MODEL_ID, # Can be a faster model
        instruction="""
You are a sharp, demanding MLB analyst and broadcast producer acting as a writing critic.
Expected in session state: `current_recap` (which is a two-host dialogue script), `game_pk`, `user_query`.

Review the `current_recap` dialogue script. Provide constructive, actionable criticism. Focus on:

- **Dialogue Flow & Engagement:**
    - Does the conversation between the hosts sound natural? Is the back-and-forth engaging?
    - Do the hosts have distinct enough 'voices' or perspectives, or do they sound too similar?
    - Is it a real discussion, or does one host merely set up the other?
- **Accuracy & Completeness (within the dialogue):**
    - Are scores, key player actions, and game sequence correctly and sufficiently detailed *as discussed by the hosts*?
- **Narrative & Engagement (of the dialogue itself):**
    - Does the *conversation* tell a compelling story? Does it have a clear narrative arc?
    - Does the dialogue capture tension, excitement, or the "story" of the game?
    - Is the language used by the hosts engaging, vivid, and journalistic?
- **Journalistic Style (of the dialogue):**
    - Does the dialogue sound like a professional sports podcast or broadcast segment?
- **Information Gaps & Opportunities for Enrichment (within the dialogue):**
    - What key information are the hosts *not* discussing that would enhance the story?
    - Are there opportunities for one host to introduce more stats or context for the other to react to?
- **Clarity & Flow (of the dialogue):**
    - Is the conversation easy to follow? Are the hosts' lines clear and concise?
- **Data Usage (by the hosts):**
    - Are stats used effectively by the hosts to support their points, or just dropped in?

If the dialogue script is excellent and requires no changes (rare!), respond ONLY with "The recap is excellent."
Otherwise, provide **specific, bulleted feedback** with clear examples of what needs improvement in the dialogue or what specific information the hosts should discuss.
        """,
        output_key="current_critique",
    )

    critique_processor_agent = LlmAgent(
        name="CritiqueProcessor",
        model=GEMINI_PRO_MODEL_ID, # Use a capable model for this multi-step reasoning
        instruction="""You are a specialized research assistant and data coordinator.
        Expected in session state: `current_critique`, `game_pk`, `user_query`.

        Your multi-step task is to process the `current_critique` and gather information for revision:

        1.  **Store the Critique:**
            *   Call the `bq_search.store_new_critique` tool.
            *   Use parameters:
                *   `critique_text` = {session.state.current_critique}
                *   `task_text` = "recap for game_pk {session.state.game_pk} based on user query '{session.state.user_query}'" (ensure `game_pk` and `user_query` are from session state).
                *   `game_pk_str` = If `session.state.game_pk` has an integer value, convert it to its string representation (e.g., if game_pk is 777930, pass "777930") and/or If `session.state.game_pk` is null/None or not present, pass an **empty string.
                *   `revision_number_str` = string representation of {session.state.revision_number} (if available, otherwise empty string).
            *   Let the result of this tool call be `critique_storage_status_json`.

        2.  **Generate Targeted Web Search Queries from Critique:**
            *   Carefully analyze the `current_critique`. Identify 1-3 key questions or information gaps highlighted by the critique that could be addressed with a web search (e.g., specific missing actual details, player's recent performance trends, injury news before the game, context of a rivalry).
            *   Formulate these as concise, effective search queries suitable for the Tavily search engine.
            *   Broader context if the critique implies it's missing (e.g., "series implications for [Team A]", "historical significance of [Team A] vs [Team B] matchup", "player [Player Name] recent performance trend").
            *   Let this list of query strings be `generated_tavily_queries`. (This is an internal thought process; you will use these queries in the next step).

        3.  **Perform Web Searches:**
            *   If you generated any `generated_tavily_queries` in step 2:
                *   For each query in `generated_tavily_queries` (max 3 queries total):
                    *   Call the `web_search.perform_web_search` tool with the `query` and `max_results=1` (or 2 if more context is needed).
                *   Collect all results. Let the combined list of web search result strings be `web_search_findings_list`. If no results, this should be an empty list or a list with "No relevant web results found."

        4.  **Perform RAG Document Search (Contextual Game Info):**
            *   Call `bq_search.search_rag_documents` using:
                *   `query_text` = {session.state.current_critique} (to find RAG docs relevant to the critique points)
                *   `game_pk_str` = string representation of {session.state.game_pk}
                *   `top_n` = 2.
            *   Let the result (a JSON string list of document contents) be `rag_findings_json_list`.

        5.  **Assemble Final JSON Output:**
            *   Construct a single JSON string as your output. This JSON object must have the following keys:
                - `"critique_storage_status"`: (string) The `critique_storage_status_json` obtained from step 1.
                - `"web_search_queries_generated"`: (list of strings) The `generated_tavily_queries` you formulated in step 2.                
                - `"web_search_findings"`: (list of strings) The `web_search_findings_list` from step 3.
                - `"rag_findings"`: (list of strings) Parse `rag_findings_json_list` from step 4 into a Python list of strings.
                - `"overall_status_message"`: (string) A brief confirmation, e.g., "Critique processed. Web and RAG searches performed based on critique."

            Example JSON output format:
            ```json
            {{
              "critique_storage_status": "{{\"status\": \"success\", \"critique_id\": \"xyz123\"}}",
              "web_search_findings": ["Tavily: Player X was indeed recovering from a minor injury before the game.", "Tavily: The rivalry dates back to a controversial playoff series in 2010."],
              "rag_findings": ["RAG: The game summary highlighted the manager's post-game comments on the team's resilience.", "RAG: Detailed play analysis shows the turning point was the 7th inning double play."],
              "overall_status_message": "Critique processed. Web and RAG searches performed to address critique points."
            }}
            ```
        Ensure your entire output is ONLY this single, valid JSON string. Do not add any explanatory text before or after the JSON.
        If a step yields no data (e.g., no web queries generated, or searches return nothing), represent that with empty lists (e.g., `"web_search_findings": []`) in the final JSON.
        """,
        tools=[
             tool for toolset_name in ["bq_search", "web_search"] for tool in loaded_mcp_tools.get(toolset_name, [])
        ],
        output_key="critique_processor_results_json"
    )

    recap_reviser_agent = LlmAgent(
        name="RecapReviser",
        model=GEMINI_PRO_MODEL_ID,
        instruction="""
You are an expert sports story editor and reviser, tasked with transforming a game recap **dialogue script** into a polished, engaging piece of sports journalism, maintaining the two-host conversational format.

Session State Expectations:
- `current_recap`: The existing version of the game recap.
- `current_critique`: The critique to address, focusing on narrative, detail, and journalistic style.
- `critique_processor_results_json`: A JSON string containing:
    - `critique_storage_status` (string)
    - `web_search_findings` (list of strings from web searches based on the critique)
    - `rag_findings` (list of strings from RAG document searches based on the critique)
- `user_query`: The original user request.
- `past_critiques_feedback` (Optional): General learnings from similar past tasks.
- `live_score_data`, `pbp_summary_data`, `comprehensive_data_json`: Core game data from the InitialRecapGenerator phase, which might be needed for cross-referencing or extracting further detail if the critique pointed to a specific factual gap not covered by new web/RAG findings.

Your Task:
1.  **Parse Research:** Parse the JSON string in `session.state.critique_processor_results_json`. Extract `web_search_findings` and `rag_findings`.
2.  **Address Critique Holistically (Maintaining Dialogue Format):**
    *   Thoroughly revise the `current_recap` (which is a dialogue script) to address *every actionable point* in `current_critique`.
    *   **Integrate Research Narratively:** Seamlessly weave in relevant information from `web_search_findings` and `rag_findings` into the *dialogue*. For example, one host might present a new finding, and the other can react or build upon it.
    *   **Enhance Storytelling & Dialogue Flow:** Elevate the language within the hosts' lines. Ensure the back-and-forth is natural and engaging.
    *   **Refine Narrative Arc:** Ensure the dialogue has a clear lead, develops the game's key moments and turning points through the hosts' discussion, and concludes effectively.
    *   **Contextualize Performances:** The hosts should discuss stats and their significance within the conversation.
3.  **Apply Stylistic Guidance (for Dialogue):**
    *   Incorporate general stylistic advice from `session.state.past_critiques_feedback` applicable to conversational sports commentary.
    *   Ensure the tone of the dialogue is appropriate.
4.  **Fulfill Original Request:** Double-check that the revised dialogue script comprehensively and engagingly addresses the `user_query`.
5.  **Clarity and Conciseness within Dialogue:** Ensure each host's lines are clear and the overall conversation flows well.
6.  **Maintain Dialogue Format:**
    *   The entire revised output MUST remain a conversation between two hosts.
    *   **Strict Alternation:** Each line of the script MUST represent one host speaking, alternating strictly.
    *   **NO Speaker Labels:** CRITICAL: Do NOT include speaker labels like "Host 1:", "Host 2:".

Your final output MUST BE ONLY the revised and improved game dialogue script text, with each speaker's line on a new line. Do not include any conversational intros, outros, or explanations about the changes you made.
        """,
        output_key="current_recap",
    )


    grammar_check_agent = LlmAgent(
        name="RecapGrammarCheck",
        model=MODEL_ID,
        instruction="""
You are a grammar and style checker for sports journalism.
Expected in session state: `current_recap`.

Review the `current_recap` for grammatical errors, awkward phrasing, and areas where the language could be more impactful or active, fitting for a professional sports recap.
Output only a JSON list of concise, actionable suggestions. If the grammar and style are excellent, output an empty list `[]` or a list containing the string "Grammar and style are good."

Example of a suggestion:
"In paragraph 2, sentence 3: 'He allowed just two hits' could be more active, e.g., 'He yielded only two hits' or 'He surrendered just two hits.'"
""",
        output_key="grammar_suggestions",
    )

    tone_check_agent = LlmAgent(
        name="RecapToneCheck",
        model=MODEL_ID,
        instruction="""
You are a tone analyzer.
Expected in session state: `current_recap`.

Analyze the tone of the `current_recap` from the perspective of a fan of the winning team or a neutral sports journalist reporting on the game's outcome.
- A dominant win (like the 8-0 example) should generally be 'positive' or at least 'neutral-positive'.
- A close, hard-fought win might be 'positive' or 'exciting'.
- A straightforward loss would likely be 'neutral' or 'negative' for the losing team's perspective, but the recap itself should aim for objective reporting where appropriate.

Consider if the language used effectively conveys the significance of the win/loss and the performances.

Output ONLY one word that best describes the overall tone: 'positive', 'negative', or 'neutral'.
""",
        output_key="tone_check_result",
    )


# --- Sub-Agents for VisualAssetWorkflowAgent ---

    entity_extractor_agent = LlmAgent(
        name="EntityExtractorForAssets",
        model=MODEL_ID, # Can be a fast model like gemini-2.0-flash
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
{
  "players": ["Shohei Ohtani", "Aaron Judge"],
  "teams": ["Los Angeles Angels", "New York Yankees"]
}

If no players or teams are found, output empty lists within the JSON object (e.g., {"players": [], "teams": []}).
Ensure your entire output is a single, valid JSON string.
        """,
        output_key="extracted_entities_json" # This will be a JSON string
    )

    # Now, update StaticAssetQueryGenerator to use this output
    static_asset_query_generator_agent = LlmAgent(
        name="StaticAssetQueryGenerator",
        model=MODEL_ID,
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
        # No tools needed for this agent, it just transforms data from session state
        output_key="static_asset_search_queries_json", # Still outputs a JSON string
    )

    # ... (rest of your LlmAgent definitions for visual_generator_mcp_caller_agent, etc.)

    static_asset_retriever_agent = LlmAgent(
    name="StaticAssetRetriever",
    model=GEMINI_PRO_MODEL_ID, # Needs to be good at instruction following and JSON manipulation
    instruction="""
You are an asset retrieval coordinator.
Expected in session state:
- 'static_asset_search_queries_json': A JSON string list of queries (e.g., '["Los Angeles Angels logo", "Willy Adames headshot"]').
- 'player_lookup_dict_json': A JSON string of a dictionary mapping player IDs to full names (e.g., '{"12345": "Willy Adames", ...}').

Your task is to process each query from 'static_asset_search_queries_json':
1.  Parse 'player_lookup_dict_json' into a Python dictionary (player_id: player_name). Create an inverse mapping (player_name_lower_case: player_id) for efficient lookup.
2.  Initialize an empty list called `found_assets_list`.
3.  For each `original_query_string` in the parsed list from 'static_asset_search_queries_json':
    a.  If the `original_query_string` contains "logo" (e.g., "Los Angeles Angels logo"):
        i.  Carefully extract the full team name (e.g., "Los Angeles Angels"). This might involve removing " logo" and trimming whitespace.
        ii. Call the `image_embedding_mcp.search_similar_images_by_text` tool with:
            - `query_text` = the extracted full team name.
            - `top_k` = 1.
            - `filter_image_type` = "logo".
        iii.The tool returns a JSON string representing a list of results. Parse this JSON string. If results are present, take the first result dictionary.
        iv. If a valid logo asset dictionary is retrieved, add `{"search_term_origin": original_query_string, **logo_asset_dict}` to `found_assets_list`. (Ensure the final dict has image_uri, type, entity_name, etc.)
    b.  If the `original_query_string` contains "headshot" (e.g., "Willy Adames headshot"):
        i.  Extract the player's full name (e.g., "Willy Adames").
        ii. Convert the extracted name to lowercase. Look up this lowercase name in your inverted player lookup map to get the `player_id`.
        iii.If a `player_id` is found:
            1. Retrieve the original casing player name using the `player_id` from the initial `player_lookup_dict`.
            2. Call the `static_retriever_mcp.get_headshot_uri_if_exists` tool with:
                - `player_id_str` = the found `player_id` (as a string).
                - `player_name_for_log` = the original casing player name.
        iv. The tool returns a JSON string. Parse it. If `image_uri` is present in the parsed dictionary, add `{"search_term_origin": original_query_string, **headshot_asset_dict}` to `found_assets_list`.
    c.  If a query is unrecognized, skip it.
4.  After processing all queries, convert `found_assets_list` into a JSON string. This is your final output.
    If 'static_asset_search_queries_json' was initially empty or no assets were successfully found and added to `found_assets_list`, output an empty JSON list string: "[]".
    """,
    tools=[
        *loaded_mcp_tools_global.get("static_retriever_mcp", []),
        *loaded_mcp_tools_global.get("image_embedding_mcp", [])
    ],
    output_key="retrieved_static_assets_json",
)

    generated_visual_prompts_agent = LlmAgent(
    name="GeneratedVisualPrompts",
    model=GEMINI_PRO_MODEL_ID,
    instruction="""
You are an assistant director analyzing an MLB game dialogue script (in session state 'current_recap') to plan visual shots for an image generation model like Imagen 3 (which filters specific names).
Identify 3-5 key moments, scenes, or actions described in the dialogue that need a generated visual.

**Critical Imagen Compatibility Rules:**
1.  **NO Player Names:** Use generic descriptions ("an MLB player", "the batter", "the pitcher", "a fielder", "the runner").
2.  **NO Team Names.**
3.  **Uniforms:** Describe generically based on home/away context if implied by the dialogue ("a player in a white home uniform", "the batter in a home jersey", "a player in a colored away uniform", "the pitcher in a gray away jersey"), or neutrally ("an MLB player's uniform"). If specific colors are mentioned in the dialogue for a generic player (e.g. "the batter in blue and orange"), use those.

**Prompt Generation Guidelines:**
*   For actions (home run, double play, strikeout), generate 1-2 distinct visual prompts representing the sequence if applicable.
*   For descriptive moments (e.g., stadium shot, manager looking tense), generate a single detailed prompt.
*   Focus on creating descriptive prompts suitable for Imagen 3. Emphasize action, emotion, setting, and relevant details like uniform descriptions based on the rules above.

Output ONLY a JSON list of 3-5 prompt strings, formatted AS A JSON STRING.
Example JSON Output (as a string): "[\\"Prompt 1...\\", \\"Prompt 2...\\"]"
If no clear visual moments, output an empty JSON list string: "[]".
If the dialogue is too short or no clear visual moments are identifiable, output an empty JSON list string: "[]".
    """,
    output_key="visual_generation_prompts_json", # Expects a JSON STRING
)

    visual_generator_mcp_caller_agent = LlmAgent( # Renamed for clarity
    name="VisualGeneratorMCPCaller",
    model=MODEL_ID,
    instruction="""
You are an image generation coordination robot.
Expected in session state:

- 'game_pk': The current game_pk (as a string or number).
- 'visual_generation_prompts_json_for_tool': A string that IS a valid JSON list of image generation prompts (e.g., "[\\"prompt1 text\\", \\"prompt2 text\\"]").
Your ONLY function is to execute the `visual_assets.generate_images_from_prompts` tool and return its raw JSON string output.
You will receive `prompts_json_string` (this is already a JSON formatted string list of prompts) from session state key 'visual_generation_prompts_json_for_tool'.
You will receive `game_pk_string` (this is already a string) from session state key 'game_pk_str_for_tool'.

Immediately call the `visual_assets.generate_images_from_prompts` tool.
Use the exact `prompts_json_string` you received for the tool's `prompts_json` parameter.
Use the exact `game_pk_string` you received for the tool's `game_pk_str` parameter.
Your entire response MUST be ONLY the direct, verbatim JSON string output from the `visual_assets_mcp.generate_images_from_prompts` tool.
Do not add any other text, explanation, or formatting.
    """,
    tools=[tool for tool in loaded_mcp_tools_global.get("visual_assets", [])],
    output_key="generated_visual_assets_uris_json", # Expects JSON string
)

    visual_critic_agent = LlmAgent(
    name="VisualCritic",
    model=MODEL_ID, # Or GEMINI_PRO_MODEL_ID
    instruction="""
You are a demanding visual producer reviewing generated images for an MLB highlight.
The primary image generator (Imagen 3) cannot use specific player/team names.

Expected in session state:
- 'current_recap': The dialogue script for context.
- 'assets_for_critique_json': JSON string list of dicts: `[{"prompt_origin": "...", "image_uri": "gs://..." or null}]`.
- 'prompts_used_for_critique_json': JSON string list of all prompts that WERE ATTEMPTED for generating the current set of images.

Critique the set of generated images based on 'assets_for_critique_json' and their corresponding 'prompts_used_for_critique_json':
1.  **Relevance to Dialogue & Prompts:** For each prompt in 'prompts_used_for_critique_json', check if its corresponding image in 'assets_for_critique_json' (if generation was successful, i.e., image_uri is not null) covers the key actions/scenes from 'current_recap' that the prompt targeted. Are there action gaps for successfully generated images?
2.  **Failures:** Note any prompts for which image generation failed (image_uri is null).
3.  **Quality/Action/Composition (for successful generations):** Are the images clear? Do they effectively convey the intended action, mood, or composition described in their 'prompt_origin', even if generic?
4.  **Suggestions for Improvement (Action/Scene Focused & Generator-Safe):** If improvements are needed (either due to failed generations or poor quality/relevance of successful ones), suggest **specific new prompts** focusing on missing *actions* or improving *composition/mood*. **Ensure suggested prompts follow the generator limitations: NO specific player names, NO specific team names.** Use descriptive generic terms.

If ALL attempted prompts resulted in successful, high-quality, relevant images OR if no prompts were provided initially, respond ONLY with "Visuals look sufficient."
Otherwise, provide concise, bullet-point feedback and **specific, generator-safe prompt suggestions** for the *next* round of image generation.
    """,
    # NO TOOLS for the critic. It only generates text.
    output_key="visual_critique_text",
)

    new_visual_prompts_from_critique_agent = LlmAgent(
    name="NewVisualPromptsFromCritique",
    model=GEMINI_PRO_MODEL_ID,
    instruction="""
You are an assistant director refining visual plans based on a critique.
Expected in session state: 'visual_critique_text'.

**Strict Image Generator Limitations (Enforce These):** NO Player Names, NO Team Names, Generic Uniforms.

Task:
Analyze `visual_critique_text`.
If critique is "Visuals look sufficient." or empty, output an empty JSON list string: `"[]"`.
Otherwise, identify visual concepts needing new/better images from the critique.
Generate a JSON list of **2-4 NEW, concise, specific prompt strings** for these concepts.
Prompts MUST be generator-safe and adhere to limitations.
Translate specific player/team mentions from critique into compliant generic descriptions. Focus on action, setting, emotion.

Example Critique: "Missing a shot of the double play. Home run image needs more excitement."
Example Output (as a JSON string):
"[\\"Dynamic action shot of two MLB fielders turning a double play...\\", \\"MLB batter celebrating enthusiastically after a home run...\\"]"

Output ONLY a JSON list string.
    """,
    output_key="new_visual_generation_prompts_json", # JSON string
)
    
    visual_asset_workflow_agent_instance = VisualAssetWorkflowAgent(
        name="VisualAssetWorkflow",
        entity_extractor=entity_extractor_agent,
        static_asset_query_generator=static_asset_query_generator_agent,
        static_asset_retriever=static_asset_retriever_agent,
        generated_visual_prompts_generator=generated_visual_prompts_agent,
        visual_generator_mcp_caller=visual_generator_mcp_caller_agent, # Matching the LlmAgent instance name
        visual_critic=visual_critic_agent,
        new_visual_prompts_creator=new_visual_prompts_from_critique_agent,
        max_visual_refinement_loops=1 # Example: 1 loop = 1 initial gen + 1 revision gen
    )

    game_recap_assistant = GameRecapAgent(
        name="game_recap_assistant",
        initial_recap_generator=initial_recap_generator_agent,
        recap_critic=recap_critic_agent,
        critique_processor=critique_processor_agent,
        recap_reviser=recap_reviser_agent,
        grammar_check=grammar_check_agent,
        tone_check=tone_check_agent,
        visual_asset_workflow=visual_asset_workflow_agent_instance, 
    )

    root_agent = LlmAgent(
        model=MODEL_ID,
        name="ai_assistant", # This is the app_name for the runner sometimes
        instruction=ROOT_AGENT_INSTRUCTION,
        sub_agents=[cocktail_agent, booking_agent, mlb_assistant, game_recap_assistant], # Added game_recap_assistant
    )
    return root_agent

# --- Agent Execution Helpers ---
async def _run_agent_and_get_response(
    runner: Runner,
    session_id: str,
    content: types.Content,
) -> List[str]:
    """
    Runs the ADK agent asynchronously for a given session and content,
    collecting and returning textual responses from the model.

    Args:
        runner: An instance of the ADK Runner.
        session_id: The unique identifier for the current session.
        content: The user's message/content to send to the agent.

    Returns:
        A list of strings, where each string is a part of the model's response.
    """
    logging.info("Running agent for session %s", session_id)
    events_async = runner.run_async(
        session_id=session_id, user_id=session_id, new_message=content
    )

    response_parts: List[str] = []
    async for event in events_async:
        try:
            if hasattr(event, "content") and event.content.role == "model":
                if hasattr(event.content, "parts") and event.content.parts:
                    part_text = getattr(event.content.parts[0], "text", None)
                    if isinstance(part_text, str) and part_text:
                        response_parts.append(part_text)
        except AttributeError as e:
            logging.warning("Could not process event attribute during agent run: %s", e)
    logging.info("Agent run finished for session %s.", session_id)
    return response_parts


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
    loaded_mcp_tools: Dict[str, Any], session_id: str, query: str
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
    response = await _run_agent_and_get_response(runner, session_id, content)
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
    websocket: WebSocket, loaded_mcp_tools: Dict[str, Any], session_id: str
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
            response_parts = await _get_runner_async(loaded_mcp_tools, session_id, text)

            if not response_parts:
                logging.info(
                    "Agent for session %s did not produce a direct text response for input: '%s'",
                    session_id,
                    text[:50],
                )
                # Consider if a specific message should be sent or just wait for next input.
                # For now, we assume if response_parts is empty, no direct message to user.
                continue

            ai_message = "\n".join(response_parts)
            await websocket.send_text(json.dumps({"message": ai_message}))
            await asyncio.sleep(0)

    except WebSocketDisconnect:
        logging.info("Client %s disconnected from run_adk_agent_async.", session_id)
    finally:
        logging.info("Agent WebSocket task ending for session %s.", session_id)

 
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

        await run_adk_agent_async(websocket, loaded_mcp_tools, session_id)

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